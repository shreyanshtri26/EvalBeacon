"""Paste-a-key live check: the key must be used once and never stored, logged, echoed or forwarded. No network."""
import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.evalbeacon import api, livecheck
from src.evalbeacon.livecheck import LiveCheckError, RateLimiter, classify, scrub

ROOT = Path(__file__).resolve().parents[1]
KEY = "sk-proj-" + "Z9y8X7w6V5u4T3s2R1q0"        # fake; the shapes real keys have
ODD_KEY = "k" + "Q8w3Er7tY2uI5oP1aS4dF6gH9jK0"      # a key no redaction pattern would recognise
GROUNDED_PREFIX = "You are a support agent. Answer ONLY"


class ProviderError(Exception):
    def __init__(self, msg, status_code=None, code=None, name=None):
        super().__init__(msg)
        self.status_code, self.code = status_code, code
        if name:
            self.__class__ = type(name, (ProviderError,), {})


def fake_client(seen_keys, fail_with=None, echo_key_in_answers=False, calls=None):
    """Grounded prompts get the retrieved context back (good answer); ungrounded ones get chatter (regression)."""
    class Completions:
        def create(self, **kw):
            if calls is not None:
                calls.append(seen_keys[-1])  # the key this client was built with, per provider request
            if fail_with:
                raise fail_with
            ctx = kw["messages"][-1]["content"]
            if kw["messages"][0]["content"].startswith(GROUNDED_PREFIX) and "Context:" in ctx:
                answer = ctx.split("Context:")[1].split("\n\nQuestion:")[0].strip() or "I am not sure; please contact support."
            else:
                answer = "Sure! Great question, it depends."
            if echo_key_in_answers:
                answer += f" key={seen_keys[-1]}"
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer), finish_reason="stop")],
                                   usage=SimpleNamespace(prompt_tokens=50, completion_tokens=20))
    return SimpleNamespace(chat=SimpleNamespace(completions=Completions()))


@pytest.fixture
def app_client(monkeypatch, tmp_path):
    """App with the live-check router; provider calls replaced; a fresh rate limiter; cwd is an empty temp dir."""
    keys: list[str] = []
    state = {"fail": None, "echo": False}
    calls: list[str] = []

    def make_client(provider, *, key=None, **kw):
        keys.append(key)
        return fake_client(keys, state["fail"], state["echo"], calls)

    monkeypatch.setattr(livecheck, "make_client", make_client)
    monkeypatch.setattr("src.evalbeacon.live.make_client", make_client)
    monkeypatch.setattr(api, "limiter", RateLimiter(max_calls=5, window_s=60))
    monkeypatch.chdir(tmp_path)
    app = FastAPI()
    app.include_router(api.router)
    api.install_error_handlers(app)
    c = TestClient(app, raise_server_exceptions=False)
    c.keys, c.state, c.tmp, c.calls = keys, state, tmp_path, calls
    return c


def post(c, **over):
    body = {"provider": "openai", "api_key": KEY, **over}
    return c.post("/api/v1/eval/live-check", json=body)


# ── the happy path ──────────────────────────────────────────────────────────
def test_live_check_runs_the_gate_with_the_pasted_key(app_client):
    r = post(app_client)
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] and d["live"] and d["provider"] == "openai" and d["model"] == "gpt-4o-mini" and d["cases"] == 10
    assert d["baseline"]["summary"]["pass_rate"] == 1.0  # grounded baseline answers from its context
    assert d["gate"]["decision"] == "fail" and d["gate"]["regressions"]  # ungrounded candidate regresses
    assert set(app_client.keys) == {KEY} and len(app_client.keys) == 3  # preflight + baseline + candidate clients
    assert set(app_client.calls) == {KEY} and len(app_client.calls) == 1 + 20  # every request used the pasted key
    assert r.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("provider,model", [("openai", "gpt-4o-mini"), ("gemini", "gemini-2.5-flash-lite"),
                                            ("deepseek", "deepseek-flash"), ("anthropic", "claude-haiku-5-5")])
def test_every_provider_defaults_to_its_cheap_model(app_client, provider, model):
    assert post(app_client, provider=provider).json()["model"] == model
    assert post(app_client, provider=provider, model="custom-model-x").json()["model"] == "custom-model-x"


def test_providers_endpoint_lists_defaults_and_model_catalogue(app_client):
    data = app_client.get("/api/v1/eval/providers").json()
    assert {p["id"]: p["default_model"] for p in data} == {"openai": "gpt-4o-mini", "gemini": "gemini-2.5-flash-lite",
                                                           "deepseek": "deepseek-flash", "anthropic": "claude-haiku-5-5"}
    for p in data:
        assert p["models"][0] == p["default_model"] and len(set(p["models"])) == len(p["models"])  # default first, no duplicates
        assert all(re.fullmatch(livecheck.MODEL_PATTERN, m) for m in p["models"])  # every suggestion is itself a valid name


@pytest.mark.parametrize("model", ["gpt-4.1-nano", "my-fine-tune_v2", "ft:gpt-4o-mini:org::abc123", "models/gemini-2.0-flash", "a" * 100])
def test_any_well_formed_custom_model_name_is_accepted(app_client, model):
    r = post(app_client, model=model)
    assert r.status_code == 200 and r.json()["model"] == model


@pytest.mark.parametrize("model", ["has space", "semi;colon", "quote\"x", "../../etc/passwd", "a" * 101, "", "model\nname", "<script>"])
def test_malformed_custom_model_names_are_rejected_without_calling_the_provider(app_client, model):
    r = post(app_client, model=model)
    assert r.status_code == 422 and not app_client.calls and KEY not in r.text


# ── the key is never stored, logged, echoed ─────────────────────────────────
def test_key_is_never_written_to_disk(app_client):
    post(app_client)
    assert not list(app_client.tmp.rglob("*"))  # no results/, history, payloads or temp files at all


def test_key_is_never_in_the_response_even_if_the_provider_echoes_it(app_client):
    app_client.state["echo"] = True
    for key in (KEY, ODD_KEY):  # ODD_KEY has a shape no redaction pattern knows: only the exact-match scrub catches it
        r = post(app_client, api_key=key)
        assert r.status_code == 200 and key not in r.text


def test_key_is_never_logged(app_client, caplog):
    with caplog.at_level(logging.DEBUG):
        post(app_client)
        post(app_client, api_key="short")  # rejected by validation
    assert KEY not in caplog.text and "short" not in caplog.text.replace("shortly", "")


def test_validation_errors_do_not_echo_the_submitted_key(app_client):
    for body in ({"provider": "nope", "api_key": KEY}, {"provider": "openai", "api_key": KEY + " has spaces"},
                 {"provider": "openai", "api_key": "x" * 500}, {"provider": "openai"}, {"provider": "openai", "api_key": KEY, "model": "bad model!"}):
        r = app_client.post("/api/v1/eval/live-check", json=body)
        assert r.status_code == 422, body
        for secret in (KEY, "x" * 500):
            assert secret not in r.text
        assert "input" not in json.dumps(r.json())  # FastAPI's default includes the rejected input; ours must not


def test_key_in_a_url_is_not_accepted(app_client):
    assert app_client.post("/api/v1/eval/live-check?api_key=" + KEY, json={"provider": "openai"}).status_code == 422
    assert app_client.get("/api/v1/eval/live-check").status_code == 405


def test_secretstr_hides_the_key_from_repr():
    m = api.LiveCheckRequest(provider="openai", api_key=KEY)
    assert KEY not in repr(m) and KEY not in str(m) and KEY not in m.model_dump_json()


def test_scrub_covers_nested_data_keys_and_json_escapes():
    data = {"a": f"x {KEY} y", "b": [{"c": KEY}], KEY: 1, "d": 5}
    out = json.dumps(scrub(data, KEY))
    assert KEY not in out and out.count("[REDACTED_KEY]") == 3


# ── preflight: clear, safe messages ─────────────────────────────────────────
@pytest.mark.parametrize("exc,status,code", [
    (ProviderError(f"Incorrect API key provided: {KEY}", 401, "invalid_api_key", "AuthenticationError"), 401, "invalid_key"),
    (ProviderError("denied", 403, None, "PermissionDeniedError"), 401, "invalid_key"),
    (ProviderError("You exceeded your current quota", 429, "insufficient_quota", "RateLimitError"), 402, "no_credit"),
    (ProviderError("Insufficient Balance", 402, None), 402, "no_credit"),
    (ProviderError("slow down", 429, "rate_limit_exceeded", "RateLimitError"), 429, "rate_limited"),
    (ProviderError("no such model", 404, None, "NotFoundError"), 404, "model_not_found"),
    (ProviderError("timed out", None, None, "APITimeoutError"), 504, "unreachable"),
    (ProviderError("connection refused", None, None, "APIConnectionError"), 504, "unreachable"),
    (ProviderError("boom", 500, None, "InternalServerError"), 502, "provider_error"),
])
def test_preflight_failures_map_to_safe_messages(app_client, exc, status, code):
    app_client.state["fail"] = exc
    r = post(app_client)
    assert r.status_code == status and r.json()["error"]["code"] == code and r.json()["stage"] == "preflight"
    assert KEY not in r.text and "Incorrect API key provided" not in r.text and "Insufficient Balance" not in r.text
    assert len(app_client.calls) == 1  # stopped after the 1-token preflight: no cases were spent


def test_classify_never_returns_provider_text():
    e = classify(ProviderError(f"secret detail {KEY}", 500), "openai")
    assert isinstance(e, LiveCheckError) and KEY not in e.message and "secret detail" not in e.message


# ── bounds ──────────────────────────────────────────────────────────────────
def test_only_whitelisted_providers_and_no_custom_base_url(app_client):
    assert post(app_client, provider="custom").status_code == 422
    r = app_client.post("/api/v1/eval/live-check", json={"provider": "openai", "api_key": KEY, "base_url": "http://169.254.169.254"})
    assert r.status_code == 200 and not any("169.254" in str(k) for k in app_client.keys)  # extra fields are ignored, not honoured


def test_rate_limit_per_client(app_client):
    codes = [post(app_client).status_code for _ in range(7)]
    assert codes[:5] == [200] * 5 and codes[5:] == [429, 429]
    r = post(app_client)
    assert r.json()["error"]["code"] == "too_many_requests" and int(r.headers["retry-after"]) >= 1 and KEY not in r.text
    n = len(app_client.calls)
    post(app_client)
    assert len(app_client.calls) == n  # a limited request makes no provider calls


def test_rate_limiter_window_and_isolation_between_clients():
    now = [0.0]
    lim = RateLimiter(max_calls=2, window_s=10, clock=lambda: now[0])
    assert lim.check("a") == 0 and lim.check("a") == 0 and lim.check("a") > 0
    assert lim.check("b") == 0  # another client is unaffected
    now[0] = 11
    assert lim.check("a") == 0  # the window slid


def test_case_count_is_capped(app_client):
    assert livecheck.MAX_CASES == 10 and livecheck.dataset_size(str(ROOT / "datasets/support_extended.jsonl")) == 10
    assert post(app_client).json()["cases"] == 10


# ── model discovery: the dropdown can be replaced by what the key can really use ──
class ModelList:
    def __init__(self, ids, fail=None):
        self.ids, self.fail = ids, fail

    def list(self):
        if self.fail:
            raise self.fail
        return SimpleNamespace(data=[SimpleNamespace(id=i) for i in self.ids])


def test_models_endpoint_filters_to_chat_models_cheapest_first(app_client, monkeypatch):
    ids = ["gpt-4o", "gpt-4o-mini", "gpt-4o-mini-2024-07-18", "gpt-4.1-nano", "text-embedding-3-small", "whisper-1", "gpt-image-1",
           "gpt-4o-mini-tts", "gpt-5-codex", "gpt-realtime", "o4-mini", "gpt-5.4-nano", "tts-1", "omni-moderation-latest"]
    monkeypatch.setattr(livecheck, "make_client", lambda provider, *, key=None, **kw: SimpleNamespace(models=ModelList(ids)))
    r = app_client.post("/api/v1/eval/models", json={"provider": "openai", "api_key": KEY})
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    models = r.json()["models"]
    assert models[:2] == ["gpt-4.1-nano", "gpt-5.4-nano"]  # cheapest tier first
    assert set(models) == {"gpt-4.1-nano", "gpt-5.4-nano", "gpt-4o-mini", "o4-mini", "gpt-4o"}  # no embeddings, audio, image, dated snapshots
    assert KEY not in r.text


def test_models_endpoint_strips_gemini_prefix_and_reports_key_problems_safely(app_client, monkeypatch):
    monkeypatch.setattr(livecheck, "make_client", lambda provider, *, key=None, **kw: SimpleNamespace(
        models=ModelList(["models/gemini-2.5-flash-lite", "models/gemini-embedding-001", "models/veo-3.1-generate-preview"])))
    assert app_client.post("/api/v1/eval/models", json={"provider": "gemini", "api_key": KEY}).json()["models"] == ["gemini-2.5-flash-lite"]
    err = ProviderError(f"Incorrect API key provided: {KEY}", 401, "invalid_api_key", "AuthenticationError")
    monkeypatch.setattr(livecheck, "make_client", lambda provider, *, key=None, **kw: SimpleNamespace(models=ModelList([], fail=err)))
    r = app_client.post("/api/v1/eval/models", json={"provider": "openai", "api_key": KEY})
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_key" and KEY not in r.text


def test_models_endpoint_validation_and_rate_limit(app_client):
    for body in ({"provider": "custom", "api_key": KEY}, {"provider": "openai", "api_key": "short"}, {"provider": "openai"}):
        r = app_client.post("/api/v1/eval/models", json=body)
        assert r.status_code == 422 and KEY not in r.text
    app_client.app.state  # noqa: B018 - the limiter is shared with the live check
    from src.evalbeacon import api as api_mod
    api_mod.limiter = RateLimiter(max_calls=1, window_s=60)
    app_client.post("/api/v1/eval/models", json={"provider": "openai", "api_key": KEY})
    assert app_client.post("/api/v1/eval/models", json={"provider": "openai", "api_key": KEY}).status_code == 429


def test_quota_exhausted_is_distinct_from_a_plain_rate_limit():
    gemini = ProviderError("Quota exceeded for quota metric 'Generate Content API requests per minute' ... limit_value 0", 429, None, "RateLimitError")
    e = classify(gemini, "gemini")
    assert e.code == "quota_exceeded" and "Gemini" in e.message and "limit of 0" in e.message
    assert classify(ProviderError("slow down", 429, "rate_limit_exceeded", "RateLimitError"), "openai").code == "rate_limited"


def test_reasoning_models_get_the_parameters_they_accept():
    from src.collectors.providers import chat_kwargs
    assert chat_kwargs("openai", "gpt-4o-mini", 200) == {"temperature": 0, "max_tokens": 200}
    for m in ("gpt-5", "gpt-5.4-nano", "o4-mini", "o3"):
        kw = chat_kwargs("openai", m, 200)
        assert "max_tokens" not in kw and "temperature" not in kw and kw["max_completion_tokens"] >= 1024  # room for hidden reasoning
    assert chat_kwargs("openai", "gpt-5.4-nano", 1, reasoning_floor=16) == {"max_completion_tokens": 16}  # cheap preflight
    assert chat_kwargs("deepseek", "deepseek-flash", 200) == {"temperature": 0, "max_tokens": 200}
    assert chat_kwargs("gemini", "gpt-5-lookalike", 200)["max_tokens"] == 200  # only OpenAI's own families are special-cased


def test_gemini_reports_a_bad_key_as_http_400_and_we_still_call_it_an_invalid_key():
    gemini = ProviderError("Error code: 400 - [{'error': {'code': 400, 'message': 'API key not valid. Please pass a valid API key.', "
                           "'status': 'INVALID_ARGUMENT', 'details': [{'reason': 'API_KEY_INVALID'}]}}]", 400, None, "BadRequestError")
    e = classify(gemini, "gemini")
    assert e.status == 401 and e.code == "invalid_key" and "Gemini rejected this key" in e.message
    # the exact text Gemini's OpenAI-compatible endpoint returned for a fake key when checked against the live service
    real = ProviderError("Error code: 400 - {'error': {'code': 400, 'message': 'Please pass a valid API key', 'status': 'INVALID_ARGUMENT'}}",
                         400, None, "BadRequestError")
    assert classify(real, "gemini").code == "invalid_key"
    # an ordinary 400 (for example a bad request body) is not blamed on the key
    assert classify(ProviderError("Unsupported parameter: max_tokens", 400, None, "BadRequestError"), "openai").code == "provider_error"
