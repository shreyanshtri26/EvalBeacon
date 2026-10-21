"""Phase 2/3/7 checks: telemetry contract, redaction boundary, failure isolation, concurrency."""
import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.trace import StatusCode
from test_instrumentor import _EXPORTER  # the single process-wide in-memory exporter

from src.evalbeacon.core import Case, Output, load_dataset, run_eval
from src.evalbeacon.redact import redact

OPENAI_KEY = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0"
GEMINI_KEY = "AIza" + "SyA1b2C3d4E5f6G7h8I9j0K1l2M3"
DATA = "datasets/support.jsonl"


def test_redact_patterns():
    s = redact(f"key {OPENAI_KEY} g={GEMINI_KEY} mail a.b@corp.io Bearer abcdefghijklmnop1234 password=hunter2")
    for leaked in (OPENAI_KEY, GEMINI_KEY, "a.b@corp.io", "abcdefghijklmnop1234", "hunter2"):
        assert leaked not in s
    assert redact(None) is None


def test_secrets_never_reach_results_or_error_messages():
    cases = [Case("c1", {}, {"must_contain": ["ok"]})]

    def leaky(_):
        return Output(f"ok, my key is {OPENAI_KEY}")

    def raises(_):
        raise RuntimeError(f"401 for key {GEMINI_KEY}")

    for agent in (leaky, raises):
        blob = json.dumps(run_eval("r", agent, cases, "v"))
        assert OPENAI_KEY not in blob and GEMINI_KEY not in blob
    # evaluators judge the raw output, so redaction cannot hide a real pass/fail
    assert run_eval("r", leaky, cases, "v")["results"][0]["status"] == "pass"


def test_capture_can_be_disabled(monkeypatch):
    monkeypatch.setenv("EVALBEACON_CAPTURE_OUTPUT", "0")
    r = run_eval("r", lambda _: Output("secret answer"), [Case("c", {}, {})], "v")
    assert r["results"][0]["output"] is None


def test_case_span_contract_and_trace_link():
    _EXPORTER.clear()
    cases, v = load_dataset(DATA)
    run = run_eval("baseline", lambda i: Output("x", 10, 5), cases[:2], v, model="gpt-4o-mini")
    spans = [s for s in _EXPORTER.get_finished_spans() if s.name == "evalbeacon.case"]
    assert len(spans) == 2
    by_trace = {format(s.context.trace_id, "032x"): s for s in spans}
    for res in run["results"]:  # every result links to its span
        a = by_trace[res["trace_id"]].attributes
        assert a["evalbeacon.case_id"] == res["id"] and a["gen_ai.usage.input_tokens"] == 10
        assert "llm.estimated_cost_usd" in a and a["evalbeacon.status"] == res["status"]


def test_error_case_marks_span_error():
    _EXPORTER.clear()
    run_eval("r", lambda _: 1 / 0, [Case("c", {}, {})], "v")
    (span,) = [s for s in _EXPORTER.get_finished_spans() if s.name == "evalbeacon.case"]
    assert span.status.status_code == StatusCode.ERROR


def test_missing_usage_is_unknown_not_zero():
    r = run_eval("r", lambda _: Output("x"), [Case("c", {}, {})], "v", model="gpt-4o-mini")["results"][0]
    assert r["cost_usd"] is None and r["cost_basis"] == "unknown"


def test_concurrent_runs_do_not_mix():
    cases, v = load_dataset(DATA)

    def go(tag):
        return run_eval(tag, lambda i: Output(f"{tag}:{i['question']}"), cases, v, workers=3)

    with ThreadPoolExecutor(4) as ex:
        runs = list(ex.map(go, [f"run{i}" for i in range(8)]))
    for run in runs:
        assert all(r["output"].startswith(run["name"] + ":") for r in run["results"])
        assert [r["id"] for r in run["results"]] == [c.id for c in cases]


# ── collector: telemetry must not corrupt or swallow application behaviour ──
def _client():
    from src.collectors.openai_collector import InstrumentedOpenAI
    return InstrumentedOpenAI(provider="deepseek", api_key="test")


def test_malformed_usage_does_not_break_successful_call():
    _EXPORTER.clear()
    resp = MagicMock(usage=None)
    with patch("openai.OpenAI") as cls:
        cls.return_value.chat.completions.create.return_value = resp
        out = _client().chat.completions.create(model="deepseek-chat", messages=[])
    assert out is resp  # application still gets its result
    (span,) = [s for s in _EXPORTER.get_finished_spans() if s.name == "deepseek chat"]
    assert span.attributes["llm.telemetry_incomplete"] is True


def test_unexpected_exception_is_recorded_and_reraised_unchanged():
    _EXPORTER.clear()
    boom = ValueError("original")
    with patch("openai.OpenAI") as cls:
        cls.return_value.chat.completions.create.side_effect = boom
        with pytest.raises(ValueError) as ei:
            _client().chat.completions.create(model="m", messages=[])
    assert ei.value is boom
    (span,) = [s for s in _EXPORTER.get_finished_spans() if s.name == "deepseek chat"]
    assert span.status.status_code == StatusCode.ERROR
