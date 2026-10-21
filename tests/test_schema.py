"""The JSON Schema must describe what the code really emits, and must reject what the contract forbids."""
import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from src.evalbeacon.cli import _cases_with_traces, build_report, main, print_gate
from src.evalbeacon.core import Case, Output, compare, load_dataset, run_eval
from src.evalbeacon.drift import detect, load_current
from src.evalbeacon.judge import JudgeReply
from src.evalbeacon.store import LocalPayloadStore

ROOT = Path(__file__).resolve().parents[1]
DATA = str(ROOT / "datasets/support.jsonl")
POLICY = str(ROOT / "configs/gate.json")
SCHEMA = json.loads((ROOT / "schemas/report.schema.json").read_text())


def validator(name: str) -> Draft202012Validator:
    return Draft202012Validator({"$schema": SCHEMA["$schema"], "$defs": SCHEMA["$defs"], "$ref": f"#/$defs/{name}"})


def errors(obj, name):
    return [f"{'/'.join(map(str, e.absolute_path))}: {e.message[:90]}" for e in validator(name).iter_errors(obj)]


def test_schema_itself_is_valid():
    Draft202012Validator.check_schema(SCHEMA)


@pytest.mark.parametrize("candidate,decision", [("candidate-good", "pass"), ("candidate-bad", "fail"), ("candidate-flaky", "fail")])
def test_real_reports_validate(candidate, decision):
    rep = build_report(DATA, candidate, POLICY)
    assert rep["gate"]["decision"] == decision and rep["schema_version"] == "1.0"
    assert errors(rep, "report") == []


def test_every_gate_decision_validates():
    cases, v = load_dataset(DATA)
    ok = build_report(DATA, "candidate-good", POLICY)
    inconclusive = compare(ok["baseline"], ok["candidate"], {"min_cases": 99})
    assert inconclusive["decision"] == "inconclusive" and errors(inconclusive, "gate") == []

    broken = run_eval("b", lambda _: 1 / 0, cases, v)  # a broken baseline is an error, not a regression
    err = compare(broken, ok["candidate"])
    assert err["decision"] == "error" and errors(err, "gate") == [] and errors(broken, "run") == []

    other = copy.deepcopy(ok["candidate"])
    other["dataset_version"] = "sha256:000000000000"
    mismatch = compare(ok["baseline"], other)
    assert mismatch["decision"] == "error" and errors(mismatch, "gate") == []


def test_unknown_values_are_null_not_zero_and_validate():
    cases, v = load_dataset(DATA)
    run = run_eval("r", lambda _: Output("return policy 30 days"), cases[:1], v)  # provider reported no usage
    r = run["results"][0]
    assert r["cost_usd"] is None and r["tokens_in"] is None and run["summary"]["total_cost_usd"] is None
    assert errors(run, "run") == []
    empty_success = run_eval("r", lambda _: 1 / 0, cases[:1], v)
    assert empty_success["summary"]["pass_rate"] == 0.0 and empty_success["summary"]["cost_per_success_usd"] is None
    assert errors(empty_success, "run") == []


def test_judge_store_and_expected_error_fields_validate(tmp_path):
    cases = [Case("j", {"question": "q"}, {"judge": {"rubric": "r", "min_score": 4}}), Case("e", {}, {"error_contains": "refused"})]

    def agent(inp):
        if "question" not in inp:
            raise PermissionError("refused by policy")
        return Output("fine", 10, 5)

    run = run_eval("r", agent, cases, "sha256:abcdef012345", model="gpt-4o-mini", store=LocalPayloadStore(tmp_path),
                   judge=lambda p: JudgeReply('{"score": 5, "rationale": "ok"}', "gpt-4o-mini", 50, 10))
    assert errors(run, "run") == [] and "judge" in run["results"][0] and run["results"][1]["expected_error"] is True


def test_drift_report_validates():
    golden = build_report(DATA, "candidate-good", POLICY)["baseline"]
    for f in ("prod_sample_stable", "prod_sample_drifted"):
        assert errors(detect(golden, load_current(ROOT / f"datasets/{f}.jsonl")), "drift") == []
    assert errors(detect(golden, {"refund-001": "x"}), "drift") == []  # inconclusive


def test_cli_output_files_validate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["demo", "--dataset", DATA, "--policy", POLICY, "--candidate", "candidate-bad"]) == 1
    for f in ("report.json",):
        assert errors(json.loads((tmp_path / "results" / f).read_text()), "report") == []
    for f in ("baseline.json", "candidate.json"):
        assert errors(json.loads((tmp_path / "results" / f).read_text()), "run") == []
    for f in (tmp_path / "results" / "history").glob("*.json"):
        assert errors(json.loads(f.read_text()), "run") == []


# ── the schema must reject contract violations ──────────────────────────────
def mutate(obj, path, value):
    o = copy.deepcopy(obj)
    cur = o
    for p in path[:-1]:
        cur = cur[p]
    cur[path[-1]] = value
    return o


@pytest.mark.parametrize("path,value", [
    (("baseline", "results", 0, "status"), "maybe"),                  # unknown status
    (("baseline", "results", 0, "trace_id"), "not-a-trace-id"),       # malformed trace id
    (("baseline", "results", 0, "cost_usd"), -1),                      # negative cost
    (("baseline", "results", 0, "reasons"), ["pass must not carry reasons"]),
    (("baseline", "summary", "pass_rate"), 1.5),
    (("baseline", "status"), "done"),
    (("baseline", "dataset_version"), "v1"),
    (("gate", "decision"), "maybe"),
    (("gate", "exit_code"), 7),
    (("schema_version"), "one"),
])
def test_schema_rejects_violations(path, value):
    rep = build_report(DATA, "candidate-good", POLICY)
    bad = mutate(rep, path if isinstance(path, tuple) else (path,), value)
    assert errors(bad, "report"), f"schema accepted a bad value at {path}"


def test_gate_decision_and_exit_code_must_agree():
    rep = build_report(DATA, "candidate-bad", POLICY)
    assert errors(mutate(rep, ("gate", "exit_code"), 0), "report")  # decision=fail but exit_code 0
    ok = build_report(DATA, "candidate-good", POLICY)
    assert errors(mutate(ok, ("gate", "reasons"), [{"gate": "x", "detail": "y", "cases": []}]), "report")  # pass with reasons


def test_missing_required_fields_are_reported():
    rep = build_report(DATA, "candidate-good", POLICY)
    for key in ("schema_version", "gate", "baseline"):
        bad = {k: v for k, v in rep.items() if k != key}
        assert errors(bad, "report")
    no_iso = copy.deepcopy(rep)
    del no_iso["candidate"]["isolation"]
    assert errors(no_iso, "report")


# ── console summary links failing cases to traces ───────────────────────────
def test_console_summary_shows_trace_ids_and_marks_missing_ones(capsys):
    rep = build_report(DATA, "candidate-bad", POLICY)
    print_gate(rep["gate"], rep["candidate"])
    out = capsys.readouterr().out
    tid = next(r["trace_id"] for r in rep["candidate"]["results"] if r["id"] == "refund-001")
    assert f"refund-001 (trace {tid[:12]}…)" in out
    print_gate(rep["gate"])  # no run supplied: honest about it instead of printing a made-up id
    assert "(no trace)" in capsys.readouterr().out
    run = copy.deepcopy(rep["candidate"])
    run["results"][0]["trace_id"] = None
    assert _cases_with_traces([run["results"][0]["id"]], run).endswith("(no trace)")
