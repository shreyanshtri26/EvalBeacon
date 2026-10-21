"""Timeout isolation, cancellation, concurrency and retry-safety contracts of the runner."""
import subprocess
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import _agents
import pytest

from src.evalbeacon.core import Case, Output, run_eval

ROOT = Path(__file__).resolve().parents[1]


def _wait_for(pred, seconds=5.0):
    end = time.time() + seconds
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_hung_agent_does_not_keep_the_process_alive():
    """Regression: abandoned agent threads were non-daemon, so the interpreter waited for them at exit."""
    code = textwrap.dedent("""
        import sys, time
        sys.path.insert(0, ".")
        from src.evalbeacon.core import Case, Output, run_eval
        def hang(_):
            time.sleep(30); return Output("late")
        r = run_eval("r", hang, [Case("c", {}, {})], "v", timeout_s=0.2)
        print(r["results"][0]["status"], r["isolation"]["still_running"])
    """)
    t0 = time.time()
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=25)
    assert out.stdout.split() == ["error", "1"], out.stderr
    assert time.time() - t0 < 12  # exits promptly instead of waiting for the 30s agent


def test_timeout_is_recorded_and_late_result_is_discarded():
    done = threading.Event()

    def slow_then_late(inp):
        if inp["q"] == "slow":
            time.sleep(0.4)
            done.set()
            return Output("late result that must never be used")
        return Output("fast")

    cases = [Case("slow", {"q": "slow"}, {"must_contain": ["late"]}),
             Case("fast", {"q": "fast"}, {"must_contain": ["fast"]})]
    run = run_eval("r", slow_then_late, cases, "v", timeout_s=0.1, workers=1)
    assert [r["status"] for r in run["results"]] == ["error", "pass"]  # the next case is unaffected
    assert "timeout" in run["results"][0]["reasons"][0] and run["status"] == "partial"
    assert run["isolation"]["timed_out_cases"] == 1 and run["isolation"]["still_running"] == 1
    assert done.wait(3)  # the abandoned thread did finish later ...
    time.sleep(0.1)
    assert run["results"][0]["status"] == "error" and run["summary"]["passed"] == 1  # ... and changed nothing


def test_cooperative_cancellation_signal(tmp_path):
    marker = tmp_path / "m.txt"
    run = run_eval("r", _agents.cooperative, [Case("c", {"path": str(marker)}, {})], "v", timeout_s=0.2)
    assert run["results"][0]["status"] == "error"
    assert _wait_for(marker.exists)
    assert marker.read_text() == "stopped cooperatively"  # cancelled() became true after the timeout


def test_process_isolation_really_terminates_the_agent(tmp_path):
    marker = tmp_path / "side_effect.txt"
    cases = [Case("c", {"sleep": 4, "path": str(marker)}, {})]
    t0 = time.time()
    run = run_eval("r", _agents.write_after_sleep, cases, "v", timeout_s=1.5, isolation="process")
    assert time.time() - t0 < 12
    assert run["results"][0]["status"] == "error" and "terminated" in run["results"][0]["reasons"][0]
    time.sleep(4.5)  # long enough for the agent to have written the file had it survived
    assert not marker.exists()
    assert run["isolation"] == {"mode": "process", "timed_out_cases": 1, "still_running": 0}


def test_process_isolation_success_error_text_and_unpicklable_agent():
    ok = run_eval("r", _agents.echo, [Case("c", {"q": "hi"}, {"must_contain": ["echo:hi"]})], "v", isolation="process")
    assert ok["results"][0]["status"] == "pass"
    refusal = Case("c", {}, {"error_contains": "refused by policy"})
    assert run_eval("r", _agents.boom, [refusal], "v", isolation="process")["results"][0]["status"] == "pass"
    lam = run_eval("r", lambda i: Output("x"), [Case("c", {}, {})], "v", isolation="process")["results"][0]
    assert lam["status"] == "error" and "picklable" in lam["reasons"][0]


def test_invalid_isolation_mode_rejected():
    with pytest.raises(ValueError, match="isolation"):
        run_eval("r", _agents.echo, [Case("c", {}, {})], "v", isolation="fork")


def test_concurrent_runs_with_timeouts_stay_isolated():
    def make(tag):
        def agent(inp):
            if inp["q"] % 3 == 0:
                time.sleep(0.3)
            return Output(f"{tag}:{inp['q']}")
        return agent

    cases = [Case(f"c{i}", {"q": i}, {}) for i in range(9)]
    with ThreadPoolExecutor(6) as ex:
        runs = list(ex.map(lambda t: run_eval(t, make(t), cases, "v", timeout_s=0.1, workers=3),
                           [f"r{i}" for i in range(6)]))
    assert len({r["run_id"] for r in runs}) == 6
    for run in runs:
        assert [x["id"] for x in run["results"]] == [c.id for c in cases]
        assert run["isolation"]["timed_out_cases"] == 3
        for x in run["results"]:
            assert x["status"] == "error" or x["output"].startswith(run["name"] + ":")  # never another run's output


def test_runner_never_retries_a_case():
    calls = []

    def flaky(inp):
        calls.append(inp["q"])
        raise ConnectionError("transient")

    run = run_eval("r", flaky, [Case("a", {"q": 1}, {}), Case("b", {"q": 2}, {})], "v")
    assert sorted(calls) == [1, 2] and run["status"] == "failed"  # one attempt each, reported as errors


def test_unexpected_agent_failure_only_affects_its_own_case():
    def agent(inp):
        if inp["q"] == 2:
            raise MemoryError("simulated")
        return Output("ok")

    run = run_eval("r", agent, [Case(str(i), {"q": i}, {}) for i in range(4)], "v")
    assert [r["status"] for r in run["results"]] == ["pass", "pass", "error", "pass"]
