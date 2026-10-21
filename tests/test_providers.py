"""Provider contract tests over a mocked HTTP transport: no network, no keys, no paid calls.

Also pins the retry contract: the SDK retries transport failures only, each retry is the same request, and the
number of upstream requests is exactly what max_retries allows.
"""
import json

import httpx
import openai
import pytest
from opentelemetry.trace import StatusCode
from test_instrumentor import _EXPORTER

from src.collectors.openai_collector import InstrumentedOpenAI
from src.evalbeacon.core import Case, Output, run_eval
from src.exporters.cost_estimator import CostEstimator

OK_BODY = {"id": "x", "object": "chat.completion", "created": 1, "model": "m",
           "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
           "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
FAST = {"retry-after-ms": "1"}  # make SDK backoff near-instant


def client(handler, provider="deepseek", max_retries=0):
    calls = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request, len(calls))

    c = InstrumentedOpenAI(provider=provider, api_key="test-key", base_url="http://provider.test/v1",
                           max_retries=max_retries, http_client=httpx.Client(transport=httpx.MockTransport(wrapped)))
    return c, calls


def chat(c):
    return c.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])


def last_span(provider="deepseek"):
    return [s for s in _EXPORTER.get_finished_spans() if s.name == f"{provider} chat"][-1]


def test_success_parses_usage_and_records_provider_span():
    _EXPORTER.clear()
    c, calls = client(lambda r, n: httpx.Response(200, json=OK_BODY), provider="gemini")
    out = chat(c)
    assert out.choices[0].message.content == "hi" and len(calls) == 1
    a = last_span("gemini").attributes
    assert a["gen_ai.system"] == "gemini" and a["gen_ai.usage.input_tokens"] == 10 and a["gen_ai.usage.output_tokens"] == 5
    assert "authorization" in {k.lower() for k in calls[0].headers}  # key sent, never logged or traced


def test_missing_usage_returns_response_and_marks_telemetry_incomplete():
    _EXPORTER.clear()
    body = {k: v for k, v in OK_BODY.items() if k != "usage"}
    c, _ = client(lambda r, n: httpx.Response(200, json=body))
    out = chat(c)
    assert out.usage is None and out.choices[0].message.content == "hi"
    assert last_span().attributes["llm.telemetry_incomplete"] is True


def test_rate_limit_without_retries_is_one_request_and_raises_unchanged():
    _EXPORTER.clear()
    c, calls = client(lambda r, n: httpx.Response(429, json={"error": {"message": "slow down", "code": "rate_limit_exceeded"}}),
                      max_retries=0)
    with pytest.raises(openai.RateLimitError):
        chat(c)
    assert len(calls) == 1
    s = last_span()
    assert s.status.status_code == StatusCode.ERROR and s.attributes["error.type"] == "rate_limit"


def test_quota_exhausted_is_distinguishable_from_a_plain_rate_limit():
    body = {"error": {"message": "You exceeded your current quota", "type": "insufficient_quota", "code": "insufficient_quota"}}
    c, _ = client(lambda r, n: httpx.Response(429, json=body))
    with pytest.raises(openai.RateLimitError) as ei:
        chat(c)
    assert ei.value.code == "insufficient_quota"


def test_retry_contract_429_then_success_uses_exactly_two_requests_and_one_span():
    _EXPORTER.clear()
    c, calls = client(lambda r, n: httpx.Response(429, headers=FAST, json={"error": {"message": "x"}}) if n == 1
                      else httpx.Response(200, json=OK_BODY), max_retries=1)
    assert chat(c).choices[0].message.content == "hi"
    assert len(calls) == 2
    assert calls[0].content == calls[1].content  # the retry is the identical request
    assert len([s for s in _EXPORTER.get_finished_spans() if s.name == "deepseek chat"]) == 1  # one logical call


def test_server_errors_stop_after_max_retries():
    c, calls = client(lambda r, n: httpx.Response(500, headers=FAST, json={"error": {"message": "boom"}}), max_retries=2)
    with pytest.raises(openai.InternalServerError):
        chat(c)
    assert len(calls) == 3  # 1 attempt + 2 retries, never more


def test_auth_failure_is_not_retried():
    c, calls = client(lambda r, n: httpx.Response(401, json={"error": {"message": "bad key"}}), max_retries=3)
    with pytest.raises(openai.AuthenticationError):
        chat(c)
    assert len(calls) == 1


def test_timeout_and_network_errors():
    def timeout(r, n):
        raise httpx.ReadTimeout("slow", request=r)

    def down(r, n):
        raise httpx.ConnectError("refused", request=r)

    c, calls = client(timeout)
    with pytest.raises(openai.APITimeoutError):
        chat(c)
    assert len(calls) == 1
    c, calls = client(down, max_retries=1)
    with pytest.raises(openai.APIConnectionError):
        chat(c)
    assert len(calls) == 2  # connection errors are the retryable class


def test_malformed_body_is_an_error_not_a_crash_of_the_run():
    _EXPORTER.clear()
    c, _ = client(lambda r, n: httpx.Response(200, content=b"<html>gateway</html>", headers={"content-type": "application/json"}))
    with pytest.raises(Exception):  # noqa: B017 — SDK raises a parse/validation error; the point is that it is raised, once
        chat(c)
    assert last_span().status.status_code == StatusCode.ERROR


def test_agent_level_provider_failure_becomes_a_case_error_with_one_request_per_case():
    def handler(r, n):
        return httpx.Response(429, json={"error": {"message": "quota", "code": "insufficient_quota"}})

    c, calls = client(handler, max_retries=0)

    def agent(inp):
        r = chat(c)
        return Output(r.choices[0].message.content)

    run = run_eval("r", agent, [Case("a", {}, {}), Case("b", {}, {})], "v", workers=1)
    assert run["status"] == "failed" and len(calls) == 2  # exactly one upstream request per case
    assert all("RateLimitError" in r["reasons"][0] for r in run["results"])


def test_unknown_model_uses_fallback_price_and_says_so():
    assert not CostEstimator.is_known("brand-new-model-9") and CostEstimator.is_known("gpt-4o-mini-2024-07-18")
    run = run_eval("r", lambda i: Output("x", 100, 50), [Case("c", {}, {})], "v", model="brand-new-model-9")
    r = run["results"][0]
    assert r["cost_usd"] is not None and "FALLBACK" in r["cost_basis"]
    known = run_eval("r", lambda i: Output("x", 100, 50), [Case("c", {}, {})], "v", model="gpt-4o-mini")["results"][0]
    assert known["cost_basis"] == "provider-usage x pricing-table estimate"


def test_json_body_is_what_the_mock_sent():  # guards the helper itself
    c, calls = client(lambda r, n: httpx.Response(200, json=OK_BODY))
    chat(c)
    assert json.loads(calls[0].content)["messages"][0]["content"] == "hi"
