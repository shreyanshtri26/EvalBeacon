"""Plan phases 4-7: run-level contract, policy validation, expected-error cases, failure isolation."""
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter

from src.evalbeacon import core
from src.evalbeacon.cli import build_report, main
from src.evalbeacon.core import Case, Output, compare, load_dataset, run_eval, validate_policy

ROOT = Path(__file__).resolve().parents[1]
DATA = str(ROOT / "datasets/support.jsonl")
POLICY = str(ROOT / "configs/gate.json")


@pytest.mark.parametrize("bad", [{"max_regresions": 0}, {"min_cases": -1}, {"max_error_rate": "x"}, {"min_cases": True}])
def test_invalid_policy_is_rejected(bad):
    with pytest.raises(ValueError):
        validate_policy(bad)


def test_cli_rejects_malformed_policy_and_missing_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bad = tmp_path / "p.json"
    bad.write_text("{not json")
    assert main(["demo", "--dataset", DATA, "--policy", str(bad)]) == 2
    bad.write_text('{"nope": 1}')
    assert main(["demo", "--dataset", DATA, "--policy", str(bad)]) == 2
    assert main(["compare", "a.json", "b.json"]) == 2


def test_storage_failure_is_a_clean_setup_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").write_text("a file where the results dir should be")
    assert main(["demo", "--dataset", DATA, "--policy", POLICY]) == 2


def test_run_level_contract():
    cases, v = load_dataset(DATA)

    def flaky(i):
        if "shipping" in i["question"]:
            raise RuntimeError("provider 503")
        return Output("return policy 30 days")

    run = run_eval("r", flaky, cases, v, model="gpt-4o-mini", revision="prompt-v2")
    assert run["status"] == "partial" and run["finished_at"] >= run["started_at"]
    assert {f["id"] for f in run["failure_summary"]} == {"shipping-001", "shipping-002"}
    assert run["config"] == {"model": "gpt-4o-mini", "revision": "prompt-v2", "dataset_version": v,
                             "evaluator_version": core.EVALUATOR_VERSION, "pricing_version": run["config"]["pricing_version"]}
    assert "unverified" in run["config"]["pricing_version"]
    assert run_eval("r", lambda _: 1 / 0, cases, v)["status"] == "failed"
    assert run_eval("r", lambda _: Output("x"), cases[:1], v)["status"] == "complete"


def test_expected_error_behaviour():
    refuses = Case("c", {}, {"error_contains": "refused"})

    def raises(_):
        raise PermissionError("tool call refused by policy")

    assert run_eval("r", raises, [refuses], "v")["results"][0]["status"] == "pass"
    wrong = run_eval("r", lambda _: 1 / 0, [refuses], "v")["results"][0]
    assert wrong["status"] == "error"  # failed, but not in the way the case expected
    silent = run_eval("r", lambda _: Output("fine"), [refuses], "v")["results"][0]
    assert silent["status"] == "fail" and "no error raised" in silent["reasons"][0]


def test_runner_calls_agent_exactly_once_per_case():
    cases, v = load_dataset(DATA)
    calls = []
    run_eval("r", lambda i: calls.append(i["question"]) or Output("x"), cases, v)
    assert sorted(calls) == sorted(c.input["question"] for c in cases)  # no retries, no duplicates


def test_evaluator_version_mismatch_is_an_error():
    r = build_report(DATA, "candidate-good", None)
    r["candidate"]["evaluator_version"] = "999"
    g = compare(r["baseline"], r["candidate"])
    assert g["decision"] == "error" and g["reasons"][0]["gate"] == "evaluator_mismatch"


def test_token_totals_reported():
    r = build_report(DATA, "candidate-bad", None)
    assert r["candidate"]["summary"]["tokens_in"] > r["baseline"]["summary"]["tokens_in"] > 0


def test_unavailable_exporter_does_not_break_the_run(monkeypatch):
    class Down(SpanExporter):
        def export(self, spans):
            raise ConnectionError("collector unreachable")

        def shutdown(self):
            pass

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(Down()))
    monkeypatch.setattr(core.trace, "get_tracer", lambda *_a, **_k: provider.get_tracer("t"))
    run = run_eval("r", lambda _: Output("ok"), [Case("c", {}, {"must_contain": ["ok"]})], "v")
    assert run["results"][0]["status"] == "pass"


def test_mock_scenarios_and_extended_dataset():
    ext = str(ROOT / "datasets/support_extended.jsonl")
    cases, _ = load_dataset(ext)
    assert len(cases) == 28
    good = build_report(ext, "candidate-good", POLICY)["gate"]
    assert good["decision"] == "pass" and good["baseline"]["pass_rate"] == 1.0
    flaky = build_report(DATA, "candidate-flaky", POLICY)["gate"]
    assert flaky["decision"] == "fail" and "error_rate" in {r["gate"] for r in flaky["reasons"]}


def test_api_report_history_sample_and_redirect(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.evalbeacon.api import router
    monkeypatch.chdir(ROOT)  # the API resolves datasets/ relative to the project root
    app = FastAPI()
    app.include_router(router)
    c = TestClient(app)
    assert c.get("/api/v1/eval/report?candidate=candidate-flaky&dataset=extended").json()["gate"]["decision"] == "fail"
    assert c.get("/api/v1/eval/report?candidate=evil").status_code == 422
    assert c.get("/api/v1/eval/report?dataset=nope").status_code == 422
    monkeypatch.chdir(tmp_path)  # no saved history here, so the labelled sample is returned
    (tmp_path / "datasets").mkdir()
    for f in ("support_extended.jsonl",):
        (tmp_path / "datasets" / f).write_text((ROOT / "datasets" / f).read_text())
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/gate.json").write_text((ROOT / "configs/gate.json").read_text())
    rows = c.get("/api/v1/eval/history").json()
    assert len(rows) == 3 and all(r["sample"] for r in rows) and {r["gate"] for r in rows} == {"pass", "fail"}
    assert c.get("/", follow_redirects=False).headers["location"] == "/ui"
