"""Chat/RAG/cost API: validation, error mapping and leak prevention. No network, no keys, no paid calls."""
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_instrumentor import _EXPORTER

from src.api import routes
from src.evalbeacon.redact import install_log_redaction

KEY = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0"


@pytest.fixture
def client():
    install_log_redaction()
    app = FastAPI()
    app.include_router(routes.router, prefix="/api/v1")
    return TestClient(app, raise_server_exceptions=False)


def fake_client(content="hi", usage=(10, 5), finish="stop", raises=None, record=None):
    class Completions:
        def create(self, **kw):
            if record is not None:
                record.append(kw)
            if raises:
                raise raises
            u = SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1]) if usage else None
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)], usage=u)
    return SimpleNamespace(chat=SimpleNamespace(completions=Completions()))


class ProviderError(Exception):
    """Looks like an SDK error: carries status_code/code and a message with a credential in it."""
    def __init__(self, msg, status_code=401, code="invalid_api_key"):
        super().__init__(msg)
        self.status_code, self.code = status_code, code


# ── validation ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("body", [
    {}, {"message": ""}, {"message": 5}, {"message": "x" * 8001}, {"message": "hi", "max_tokens": 0},
    {"message": "hi", "max_tokens": 99999}, {"message": "hi", "provider": "bad name!"}, {"message": "hi", "model": "x" * 101},
    {"message": "hi", "system_prompt": "s" * 4001}, {"message": ["not", "a", "string"]},
])
def test_chat_rejects_invalid_bodies(client, body):
    r = client.post("/api/v1/chat", json=body)
    assert r.status_code == 422 and "detail" in r.json()


def test_chat_rejects_malformed_json_and_wrong_content_type(client):
    assert client.post("/api/v1/chat", content=b"{not json", headers={"content-type": "application/json"}).status_code == 422
    assert client.post("/api/v1/chat", content=b"message=hi", headers={"content-type": "text/plain"}).status_code == 422


@pytest.mark.parametrize("body", [{}, {"query": ""}, {"query": "q", "top_k": 0}, {"query": "q", "top_k": 21}, {"query": "q" * 2001}])
def test_rag_rejects_invalid_bodies(client, body):
    assert client.post("/api/v1/rag/query", json=body).status_code == 422


@pytest.mark.parametrize("body", [{"model": "m", "input_tokens": -1, "output_tokens": 1}, {"model": "m", "input_tokens": "x", "output_tokens": 1},
                                  {"model": "", "input_tokens": 1, "output_tokens": 1}, {"input_tokens": 1, "output_tokens": 1}])
def test_cost_estimate_validation(client, body):
    assert client.post("/api/v1/cost/estimate", json=body).status_code == 422


def test_monthly_projection_validation_and_success(client):
    assert client.get("/api/v1/cost/monthly-projection?daily_requests=-5").status_code == 422
    assert client.get("/api/v1/cost/monthly-projection?model=bad%20name").status_code == 422
    ok = client.get("/api/v1/cost/monthly-projection?model=gpt-4o-mini&daily_requests=10").json()
    assert ok["projected_monthly_cost_usd"] > 0


def test_unknown_model_is_flagged_not_hidden(client):
    r = client.post("/api/v1/cost/estimate", json={"model": "gpt-4o-mini", "input_tokens": 1000, "output_tokens": 1000}).json()
    u = client.post("/api/v1/cost/estimate", json={"model": "mystery-model", "input_tokens": 1000, "output_tokens": 1000}).json()
    assert r["known_model"] is True and u["known_model"] is False and u["estimated_cost_usd"] > 0


# ── success paths ───────────────────────────────────────────────────────────
def test_chat_success_with_provider_default_model(client, monkeypatch):
    calls = []
    monkeypatch.setattr(routes, "make_client", lambda provider: fake_client(record=calls))
    r = client.post("/api/v1/chat", json={"message": "hello", "provider": "deepseek"})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "deepseek-flash" and body["input_tokens"] == 10 and body["estimated_cost_usd"] > 0
    assert calls[0]["messages"][-1] == {"role": "user", "content": "hello"} and calls[0]["max_tokens"] == 1024


def test_chat_missing_usage_is_null_not_zero(client, monkeypatch):
    monkeypatch.setattr(routes, "make_client", lambda provider: fake_client(usage=None))
    body = client.post("/api/v1/chat", json={"message": "hello"}).json()
    assert body["input_tokens"] is None and body["output_tokens"] is None and body["estimated_cost_usd"] is None


def test_chat_empty_choices_does_not_crash(client, monkeypatch):
    def empty():
        c = fake_client()
        c.chat.completions.create = lambda **kw: SimpleNamespace(choices=[], usage=None)
        return c
    monkeypatch.setattr(routes, "make_client", lambda provider: empty())
    body = client.post("/api/v1/chat", json={"message": "hello"}).json()
    assert body["response"] == "" and body["finish_reason"] == "unknown"


# ── errors: mapping and leak prevention ─────────────────────────────────────
def test_unknown_provider_and_missing_key_are_400(client, monkeypatch):
    r = client.post("/api/v1/chat", json={"message": "hi", "provider": "nope"})
    assert r.status_code == 400 and "unknown provider" in r.json()["detail"]

    def no_key(provider):
        raise ValueError("DEEPSEEK_API_KEY is not set (add it to .env)")
    monkeypatch.setattr(routes, "make_client", no_key)
    r = client.post("/api/v1/chat", json={"message": "hi", "provider": "deepseek"})
    assert r.status_code == 400 and "DEEPSEEK_API_KEY is not set" in r.json()["detail"]


def test_provider_failure_is_502_with_sanitised_message(client, monkeypatch):
    err = ProviderError(f"Incorrect API key provided: {KEY}. contact alice@corp.io request_id=abc-123")
    monkeypatch.setattr(routes, "make_client", lambda provider: fake_client(raises=err))
    r = client.post("/api/v1/chat", json={"message": "hi"})
    assert r.status_code == 502
    detail = r.json()["detail"]
    assert detail == "LLM provider error (ProviderError, status 401, code invalid_api_key)"
    assert KEY not in r.text and "alice@corp.io" not in r.text and "abc-123" not in r.text


def test_provider_failure_is_not_leaked_through_logs(client, monkeypatch, caplog):
    err = ProviderError(f"bad key {KEY}")
    monkeypatch.setattr(routes, "make_client", lambda provider: fake_client(raises=err))
    with caplog.at_level("ERROR"):
        client.post("/api/v1/chat", json={"message": "hi"})
    rec = [r for r in caplog.records if "Chat request failed" in r.getMessage()][-1]
    assert KEY not in caplog.text and KEY not in str(rec.__dict__) and rec.exc_info is None


def test_unexpected_status_and_weird_code_are_not_echoed(client, monkeypatch):
    err = ProviderError("boom", status_code=500, code=f"<script>{KEY}</script>")  # a hostile code must not be echoed
    monkeypatch.setattr(routes, "make_client", lambda provider: fake_client(raises=err))
    r = client.post("/api/v1/chat", json={"message": "hi"})
    assert r.status_code == 502 and "script" not in r.text and KEY not in r.text and "status 500" in r.json()["detail"]
    monkeypatch.setattr(routes, "make_client", lambda provider: fake_client(raises=TimeoutError("socket timed out")))
    assert client.post("/api/v1/chat", json={"message": "hi"}).json()["detail"] == "LLM provider error (TimeoutError)"


# ── RAG ─────────────────────────────────────────────────────────────────────
def test_rag_success_and_spans_do_not_contain_the_query(client, monkeypatch):
    _EXPORTER.clear()
    monkeypatch.setattr(routes, "make_client", lambda provider: fake_client(content="answer"))
    q = "UNIQUE-QUERY-TEXT about my account"
    r = client.post("/api/v1/rag/query", json={"query": q, "top_k": 2})
    assert r.status_code == 200 and r.json()["answer"] == "answer" and len(r.json()["sources"]) == 2
    attrs = json.dumps({k: str(v) for s in _EXPORTER.get_finished_spans() for k, v in s.attributes.items()})
    assert "UNIQUE-QUERY-TEXT" not in attrs and "rag.query_sha256" in attrs and "rag.query_chars" in attrs


def test_rag_provider_failure_is_sanitised_and_missing_usage_ok(client, monkeypatch):
    monkeypatch.setattr(routes, "make_client", lambda provider: fake_client(raises=ProviderError(f"x {KEY}", 429, "rate_limit_exceeded")))
    r = client.post("/api/v1/rag/query", json={"query": "q"})
    assert r.status_code == 502 and KEY not in r.text and "status 429" in r.json()["detail"]
    monkeypatch.setattr(routes, "make_client", lambda provider: fake_client(usage=None))
    ok = client.post("/api/v1/rag/query", json={"query": "q"}).json()
    assert ok["token_usage"] == {"input": None, "output": None}
    assert client.post("/api/v1/rag/query", json={"query": "q", "provider": "nope"}).status_code == 400
