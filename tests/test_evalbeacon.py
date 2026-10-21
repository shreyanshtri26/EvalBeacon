"""Tests for EvalBeacon: dataset, evaluators, runner, gate, CLI exit codes."""
import json
import time
from pathlib import Path

import pytest

from src.evalbeacon.cli import build_report, main
from src.evalbeacon.core import Output, compare, evaluate, load_dataset, run_eval

DATA = "datasets/support.jsonl"


def test_dataset_hash_is_stable_and_validates(tmp_path):
    cases, v1 = load_dataset(DATA)
    assert len(cases) == 10 and load_dataset(DATA)[1] == v1
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id":"a","input":{},"expected":{}}\n{"id":"a","input":{},"expected":{}}\n')
    with pytest.raises(ValueError, match="duplicate id"):
        load_dataset(bad)
    bad.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_dataset(bad)
    bad.write_text("not json\n")
    with pytest.raises(ValueError, match=":1:"):
        load_dataset(bad)


def test_evaluators():
    o = Output('{"a": 1}', tool_calls=["t"])
    assert evaluate({"json_fields": ["a"], "tool_calls": ["t"], "must_contain": ["A"]}, o) == []
    assert len(evaluate({"json_fields": ["b"], "must_not_contain": ["a"], "regex": "zzz", "exact": "x"}, o)) == 4
    assert evaluate({"json_fields": ["a"]}, Output("nope"))[0].startswith("json_fields: output is not valid")


def test_runner_converts_errors_and_timeouts_to_results():
    cases, v = load_dataset(DATA)
    cases = cases[:2]

    def boom(_):
        raise RuntimeError("provider down")

    def slow(_):
        time.sleep(0.5)
        return Output("x")

    assert [r["status"] for r in run_eval("b", boom, cases, v)["results"]] == ["error", "error"]
    run = run_eval("s", slow, cases, v, timeout_s=0.05)
    assert "timeout" in run["results"][0]["reasons"][0]
    assert run["summary"]["total_cost_usd"] is None  # unknown stays unknown, not 0


def test_runs_are_repeatable():
    a = build_report(DATA, "candidate-bad", None)["gate"]
    b = build_report(DATA, "candidate-bad", None)["gate"]
    assert a["regressions"] == b["regressions"] and a["decision"] == b["decision"]


def test_good_candidate_passes_and_bad_candidate_fails_for_expected_reason():
    good = build_report(DATA, "candidate-good", "configs/gate.json")["gate"]
    assert good["decision"] == "pass" and good["exit_code"] == 0
    bad = build_report(DATA, "candidate-bad", "configs/gate.json")["gate"]
    assert bad["decision"] == "fail"
    gates = {r["gate"] for r in bad["reasons"]}
    assert {"pass_rate", "regressions", "cost"} <= gates
    assert bad["regressions"]  # linked to case ids


def test_gate_edge_cases():
    r = build_report(DATA, "candidate-good", None)
    assert compare(r["baseline"], r["candidate"], {"min_cases": 99})["decision"] == "inconclusive"
    r["candidate"]["dataset_version"] = "other"
    assert compare(r["baseline"], r["candidate"])["decision"] == "error"


def test_broken_baseline_is_setup_error_not_regression():
    cases, v = load_dataset(DATA)

    def boom(_):
        raise RuntimeError("402 insufficient balance")

    ok = build_report(DATA, "candidate-good", None)["candidate"]
    g = compare(run_eval("b", boom, cases, v), ok)
    assert g["decision"] == "error" and g["exit_code"] == 2 and "insufficient balance" in g["reasons"][0]["detail"]


def test_cli_exit_codes(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    ds, pol = str(root / DATA), str(root / "configs/gate.json")
    monkeypatch.chdir(tmp_path)  # results/ is written to cwd
    assert main(["demo", "--dataset", ds, "--policy", pol, "--candidate", "candidate-bad"]) == 1
    assert main(["demo", "--dataset", ds, "--policy", pol, "--candidate", "candidate-good"]) == 0
    assert main(["demo", "--dataset", "missing.jsonl", "--policy", pol]) == 2
    assert json.loads((tmp_path / "results" / "report.json").read_text())["gate"]["decision"] == "pass"
