"""The scripts CI relies on, tested for the semantics the workflow depends on."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.evalbeacon.cli import build_report

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
DATA, POLICY = str(ROOT / "datasets/support.jsonl"), str(ROOT / "configs/gate.json")
CLI = [PY, "-m", "src.evalbeacon.cli"]


def run(*args, cwd=ROOT):
    return subprocess.run([PY, *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)


# ── expect_exit: an expected failure is success, anything else is failure with a diagnosis ──
def test_expect_exit_matches_and_mismatches():
    ok = run("scripts/expect_exit.py", "3", "--", PY, "-c", "raise SystemExit(3)")
    assert ok.returncode == 0 and "as expected" in ok.stdout
    bad = run("scripts/expect_exit.py", "1", "--", PY, "-c", "raise SystemExit(0)")
    assert bad.returncode == 1 and "expected exit 1" in bad.stderr and "got 0" in bad.stderr


def test_expect_exit_does_not_mistake_a_setup_error_for_a_regression():
    r = run("scripts/expect_exit.py", "1", "--", PY, "-c", "raise SystemExit(2)")
    assert r.returncode == 1 and "SETUP OR PROVIDER ERROR" in r.stderr
    assert "instead of treating this as a regression" in r.stderr


def test_expect_exit_usage_errors():
    assert run("scripts/expect_exit.py").returncode == 64
    assert run("scripts/expect_exit.py", "x", "--", "true").returncode == 64
    assert run("scripts/expect_exit.py", "1", "--").returncode == 64


# ── the commands the workflow runs, with the exit codes it expects ──────────
@pytest.mark.parametrize("candidate,code", [("candidate-good", 0), ("candidate-bad", 1), ("candidate-flaky", 1)])
def test_workflow_gate_commands(tmp_path, candidate, code):
    r = subprocess.run([*CLI, "demo", "--dataset", DATA, "--policy", POLICY, "--candidate", candidate],
                       cwd=tmp_path, capture_output=True, text=True, timeout=120, env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
    assert r.returncode == code, r.stderr[-300:]
    assert (tmp_path / "results" / "report.json").exists()


def test_setup_failure_exits_2_and_is_not_a_gate_result(tmp_path):
    env = {**__import__("os").environ, "PYTHONPATH": str(ROOT)}
    for extra in (["--policy", str(tmp_path / "missing.json")], ["--dataset", str(tmp_path / "nope.jsonl")]):
        r = subprocess.run([*CLI, "demo", *extra], cwd=tmp_path, capture_output=True, text=True, timeout=60, env=env)
        assert r.returncode == 2 and "GATE" not in r.stdout


# ── summary and validation scripts ──────────────────────────────────────────
def test_ci_summary_lists_reasons_cases_and_traces(tmp_path):
    rep = build_report(DATA, "candidate-bad", POLICY)
    f = tmp_path / "report.json"
    f.write_text(json.dumps(rep))
    r = run("scripts/ci_summary.py", str(f))
    assert r.returncode == 0 and "EvalBeacon gate: FAIL" in r.stdout and "**pass_rate**" in r.stdout and "`refund-001`" in r.stdout
    assert ("trace `" in r.stdout or "no trace" in r.stdout) and "Policy" in r.stdout


def test_ci_summary_never_fails_the_job_on_a_bad_file(tmp_path):
    r = run("scripts/ci_summary.py", str(tmp_path / "missing.json"))
    assert r.returncode == 0 and "summary unavailable" in r.stdout


def test_validate_report_script(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(build_report(DATA, "candidate-good", POLICY)))
    assert run("scripts/validate_report.py", str(good)).returncode == 0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": "1.0", "baseline": {}, "candidate": {}, "gate": {}}))
    r = run("scripts/validate_report.py", str(bad))
    assert r.returncode == 1 and "FAIL" in r.stdout
    assert run("scripts/validate_report.py", str(tmp_path / "nope.json")).returncode == 1


def test_workflow_yaml_is_valid_and_wires_the_expected_steps():
    yaml = pytest.importorskip("yaml")
    wf = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    triggers = wf.get("on") or wf[True]  # PyYAML parses the bare key `on` as boolean True
    assert {"push", "pull_request", "schedule"} <= set(triggers)
    text = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    for needle in ("ruff check", "mypy", "pytest", "expect_exit.py 1", "candidate-good", "candidate-bad", "candidate-flaky",
                   "validate_report.py", "ci_summary.py", "upload-artifact", "if: always()"):
        assert needle in text, f"workflow is missing: {needle}"
    for job in wf["jobs"].values():
        assert job["steps"], "empty job"
