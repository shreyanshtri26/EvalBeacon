"""EvalBeacon core: dataset, deterministic evaluators, runner, comparison, release gate.

Everything here is pure Python and deterministic; no provider key is needed.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import multiprocessing
import os
import re
import statistics
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from src.evalbeacon.judge import JudgeFn, judge_case
from src.evalbeacon.redact import redact
from src.evalbeacon.similarity import EmbedFn, similarity
from src.evalbeacon.store import PayloadStore
from src.exporters.cost_estimator import PRICING_VERSION, CostEstimator

EVALUATOR_VERSION = "1"
SCHEMA_VERSION = "1.0"  # schemas/report.schema.json; additive changes bump the minor number


# ── Dataset ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Case:
    id: str
    input: dict
    expected: dict
    metadata: dict = field(default_factory=dict)


def load_dataset(path: str | Path) -> tuple[list[Case], str]:
    """Load JSONL cases; return (cases, content-hash version). Raises ValueError with line numbers."""
    raw = Path(path).read_bytes()
    cases, seen = [], set()
    for n, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            cid, inp, exp = d["id"], d["input"], d["expected"]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise ValueError(f"{path}:{n}: invalid case ({e!r})") from e
        if cid in seen:
            raise ValueError(f"{path}:{n}: duplicate id {cid!r}")
        if not isinstance(inp, dict) or not isinstance(exp, dict):
            raise ValueError(f"{path}:{n}: 'input' and 'expected' must be objects")
        seen.add(cid)
        cases.append(Case(cid, inp, exp, d.get("metadata", {})))
    if not cases:
        raise ValueError(f"{path}: dataset is empty")
    return cases, "sha256:" + hashlib.sha256(raw).hexdigest()[:12]


# ── Agent output ────────────────────────────────────────────────────────────
@dataclass
class Output:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    tool_calls: list[str] = field(default_factory=list)
    trace_id: str | None = None


Agent = Callable[[dict], Output]


# ── Evaluators (deterministic) ──────────────────────────────────────────────
def evaluate(expected: dict, out: Output, embed_fn: EmbedFn | None = None) -> list[str]:
    """Return a list of failure reasons; empty means pass."""
    text, why = out.text, []
    low = text.lower()
    if "exact" in expected and text.strip() != expected["exact"].strip():
        why.append("exact: output differs")
    why += [f"must_contain: missing {t!r}" for t in expected.get("must_contain", []) if t.lower() not in low]
    why += [f"must_not_contain: found {t!r}" for t in expected.get("must_not_contain", []) if t.lower() in low]
    if "regex" in expected and not re.search(expected["regex"], text):
        why.append(f"regex: no match for {expected['regex']!r}")
    if "json_fields" in expected:
        try:
            obj = json.loads(text)
            why += [f"json_fields: missing {f!r}" for f in expected["json_fields"] if f not in obj]
        except (json.JSONDecodeError, TypeError):
            why.append("json_fields: output is not valid JSON")
    if "similar_to" in expected:
        spec = expected["similar_to"]
        score = similarity(spec["text"], text, embed_fn)
        if score < spec.get("min", 0.6):
            why.append(f"similar_to: similarity {score} < {spec.get('min', 0.6)}")
    if "error_contains" in expected:  # the case expects the agent to fail; reaching here means it did not
        why.append(f"error_contains: no error raised (expected {expected['error_contains']!r})")
    if "tool_calls" in expected and out.tool_calls != expected["tool_calls"]:
        why.append(f"tool_calls: expected {expected['tool_calls']}, got {out.tool_calls}")
    return why


# ── Isolation ───────────────────────────────────────────────────────────────
class CaseTimeoutError(TimeoutError):
    """The agent did not finish within the per-case timeout."""


_CANCEL: contextvars.ContextVar[threading.Event | None] = contextvars.ContextVar("evalbeacon_cancel", default=None)


def cancelled() -> bool:
    """Cooperative cancellation: long-running agents can poll this and stop once their case timed out."""
    ev = _CANCEL.get()
    return bool(ev and ev.is_set())


def _call_thread(agent: Agent, inp: dict, timeout: float, tag: str = "") -> tuple[Output, None]:
    """Run the agent on its own *daemon* thread. On timeout the case is abandoned, its late result is
    discarded and cancelled() turns true; the thread itself cannot be killed (Python has no safe way to)."""
    box: dict[str, Any] = {}
    cancel = threading.Event()
    ctx = contextvars.copy_context()

    def target() -> None:
        _CANCEL.set(cancel)
        try:
            box["out"] = agent(inp)
        except BaseException as e:  # noqa: BLE001 — surfaced to the caller below
            box["err"] = e

    t = threading.Thread(target=lambda: ctx.run(target), daemon=True, name=f"evalbeacon-case-{tag}")
    t.start()
    t.join(timeout)
    if t.is_alive():
        cancel.set()
        raise CaseTimeoutError(f"timeout after {timeout}s")
    if "err" in box:
        raise box["err"]
    return box["out"], None


def _proc_target(conn, agent: Agent, inp: dict) -> None:
    try:
        conn.send(("ok", agent(inp)))
    except BaseException as e:  # noqa: BLE001 — exceptions may not pickle, send text
        conn.send(("err", f"{type(e).__name__}: {e}"))


def _call_process(agent: Agent, inp: dict, timeout: float, tag: str = "") -> tuple[Output, None]:
    """Run the agent in a spawned child process that is terminated on timeout (real cancellation).
    The agent and its input/output must be picklable (a module-level function); spans created inside the
    child are not exported to this process."""
    mp = multiprocessing.get_context("spawn")
    parent, child = mp.Pipe(duplex=False)
    proc = mp.Process(target=_proc_target, args=(child, agent, inp), daemon=True)
    try:
        proc.start()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"agent cannot run in a subprocess (must be a picklable module-level function): {e}") from e
    child.close()
    try:
        if not parent.poll(timeout):
            proc.terminate()
            proc.join(2)
            if proc.is_alive():
                proc.kill()
                proc.join(2)
            raise CaseTimeoutError(f"timeout after {timeout}s (agent process terminated)")
        try:
            kind, val = parent.recv()
        except EOFError as e:
            raise RuntimeError("agent process exited without returning a result") from e
        proc.join(5)
        if kind == "err":
            raise RuntimeError(val)
        return val, None
    finally:
        parent.close()


# ── Runner ──────────────────────────────────────────────────────────────────
def run_eval(name: str, agent: Agent, cases: list[Case], dataset_version: str, *,
             model: str = "unknown", timeout_s: float = 30.0, workers: int = 4,
             revision: str = "unversioned", store: PayloadStore | None = None, judge: JudgeFn | None = None,
             embed_fn: EmbedFn | None = None, run_id: str | None = None, isolation: str = "thread") -> dict:
    """Run every case; errors and timeouts become failed results, never exceptions.

    isolation="thread" (default): each case runs on a daemon thread; a timed-out thread cannot be killed but is
    abandoned, its result discarded and cancelled() set. isolation="process": each case runs in a child process
    that is terminated on timeout (agent must be picklable). The runner itself never retries a case."""
    if isolation not in ("thread", "process"):
        raise ValueError(f"isolation must be 'thread' or 'process', got {isolation!r}")
    call = _call_thread if isolation == "thread" else _call_process
    timed_out = 0
    run_id = run_id or uuid.uuid4().hex[:12]
    tracer = trace.get_tracer("evalbeacon")
    capture = os.getenv("EVALBEACON_CAPTURE_OUTPUT", "1") != "0"  # "0" stores no model output at all

    def one(c: Case) -> dict:
        t0 = time.perf_counter()
        r: dict[str, Any] = {"id": c.id, "status": "pass", "reasons": [], "category": c.metadata.get("category")}
        raw_text = ""
        with tracer.start_as_current_span("evalbeacon.case") as span:
            ctx = span.get_span_context()
            r["trace_id"] = f"{ctx.trace_id:032x}" if ctx.is_valid else None
            span.set_attribute("evalbeacon.run", name)
            span.set_attribute("evalbeacon.case_id", c.id)
            span.set_attribute("gen_ai.request.model", model)
            try:
                # copy_context so the agent's own spans nest under this case span
                out, _ = call(agent, c.input, timeout_s, run_id)
                raw_text = out.text
                r["reasons"] = evaluate(c.expected, out, embed_fn)  # judged on the raw output, before redaction
                if "judge" in c.expected:  # optional; deterministic checks above still have to pass
                    if judge is None:
                        raise RuntimeError("case requires an LLM judge but none is configured")
                    j = judge_case(judge, str(c.input.get("question", c.input)), redact(out.text), c.expected["judge"])
                    r["judge"] = j
                    if not j["passed"]:
                        r["reasons"].append(f"judge: score {j['score']} < {j['min_score']} ({j['rationale']})")
                r["status"] = "fail" if r["reasons"] else "pass"
                r["output"] = redact(out.text[:500]) if capture else None
                r["tokens_in"], r["tokens_out"] = out.input_tokens, out.output_tokens
                # cost stays None (unknown) when the provider reported no usage
                r["cost_usd"] = (CostEstimator.estimate(model, out.input_tokens, out.output_tokens)
                                 if out.input_tokens is not None and out.output_tokens is not None else None)
                r["cost_basis"] = ("unknown" if r["cost_usd"] is None else
                                   "provider-usage x pricing-table estimate" if CostEstimator.is_known(model)
                                   else "provider-usage x FALLBACK price (model not in pricing table)")
                if out.input_tokens is not None:
                    span.set_attribute("gen_ai.usage.input_tokens", out.input_tokens)
                    span.set_attribute("gen_ai.usage.output_tokens", out.output_tokens or 0)
                if r["cost_usd"] is not None:
                    span.set_attribute("llm.estimated_cost_usd", r["cost_usd"])
            except CaseTimeoutError as e:
                nonlocal timed_out
                timed_out += 1
                r.update(status="error", reasons=[str(e)])
            except Exception as e:  # noqa: BLE001 — a failing agent must not kill the run
                msg = f"{type(e).__name__}: {e}"
                want = c.expected.get("error_contains")
                if want and want.lower() in msg.lower():  # expected error behaviour: a pass, not an error
                    r.update(status="pass", reasons=[], output=redact(msg[:500]) if capture else None,
                             expected_error=True)
                else:
                    r.update(status="error", reasons=[redact(msg)])
            r["reasons"] = [redact(x) for x in r["reasons"]]
            if store is not None and capture:  # full payload goes to the store (redacted there), not onto spans
                try:
                    r["payload_ref"] = store.put(f"{run_id}/{c.id}", {
                        "input": c.input, "expected": c.expected, "output": raw_text, "trace_id": r["trace_id"]})
                except Exception as e:  # noqa: BLE001 — a storage outage must not fail the evaluation
                    r["payload_ref"], r["store_error"] = None, redact(f"{type(e).__name__}: {e}")
            r["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            span.set_attribute("evalbeacon.status", r["status"])
            span.set_attribute("evalbeacon.latency_ms", r["latency_ms"])
            if r["status"] == "error":
                span.set_status(Status(StatusCode.ERROR, r["reasons"][0][:200]))
        return r

    started = time.time()
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(one, cases))  # map preserves case order → stable output
    summary = summarize(results)
    errored = [r for r in results if r["status"] == "error"]
    return {"schema_version": SCHEMA_VERSION, "run_id": run_id, "name": name, "model": model, "dataset_version": dataset_version,
            "evaluator_version": EVALUATOR_VERSION, "started_at": started, "finished_at": time.time(),
            # complete: no errors · partial: some cases errored · failed: every case errored
            "status": "failed" if errored and len(errored) == len(results) else "partial" if errored else "complete",
            "failure_summary": [{"id": r["id"], "reason": r["reasons"][0][:200]} for r in errored],
            # what identifies this configuration (plan phase 5)
            "config": {"model": model, "revision": revision, "dataset_version": dataset_version,
                       "evaluator_version": EVALUATOR_VERSION, "pricing_version": PRICING_VERSION},
            # still_running: timed-out agent threads of THIS run that are alive now (threads cannot be killed)
            "isolation": {"mode": isolation, "timed_out_cases": timed_out,
                          "still_running": sum(t.is_alive() and t.name == f"evalbeacon-case-{run_id}"
                                               for t in threading.enumerate())},
            "results": results, "summary": summary}


def _pct(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(round(p * (len(s) - 1))))], 2)


def summarize(results: list[dict]) -> dict:
    n = len(results)
    passed = sum(r["status"] == "pass" for r in results)
    errors = sum(r["status"] == "error" for r in results)
    costs = [r["cost_usd"] for r in results if r.get("cost_usd") is not None]
    lat = [r["latency_ms"] for r in results]
    total_cost = round(sum(costs), 8) if costs else None
    return {
        "cases": n, "passed": passed, "errors": errors,
        "pass_rate": round(passed / n, 4) if n else None,
        "error_rate": round(errors / n, 4) if n else None,
        "latency_p50_ms": _pct(lat, 0.5), "latency_p95_ms": _pct(lat, 0.95),
        "latency_mean_ms": round(statistics.fmean(lat), 2) if lat else None,
        "judged_cases": sum("judge" in r for r in results),
        "judge_cost_usd": round(sum(r["judge"]["cost_usd"] or 0 for r in results if "judge" in r), 8),
        "tokens_in": sum(r.get("tokens_in") or 0 for r in results),
        "tokens_out": sum(r.get("tokens_out") or 0 for r in results),
        "total_cost_usd": total_cost,
        "cost_per_success_usd": round(total_cost / passed, 8) if total_cost is not None and passed else None,
        "cost_known_cases": len(costs),
    }


# ── Compare + release gate ──────────────────────────────────────────────────
DEFAULT_POLICY = {
    "min_cases": 5,                 # below this the gate is inconclusive, not passing
    "max_pass_rate_drop": 0.05,     # absolute drop vs baseline
    "max_error_rate": 0.10,
    "max_p95_latency_increase_pct": 50.0,
    "max_cost_increase_pct": 25.0,
    "max_regressions": 0,           # cases that passed in baseline but fail in candidate
}

def validate_policy(policy: dict) -> dict:
    """Merge with defaults; reject unknown keys and non-numeric or negative values."""
    unknown = set(policy) - set(DEFAULT_POLICY)
    if unknown:
        raise ValueError(f"unknown gate policy keys: {sorted(unknown)}; valid: {sorted(DEFAULT_POLICY)}")
    for k, v in policy.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
            raise ValueError(f"gate policy {k!r} must be a non-negative number, got {v!r}")
    return {**DEFAULT_POLICY, **policy}


EXIT_CODES = {"pass": 0, "fail": 1, "error": 2, "inconclusive": 3}


def compare(baseline: dict, candidate: dict, policy: dict | None = None) -> dict:
    """Compare two runs on the same dataset; return a gate decision with reasons."""
    p = validate_policy(policy or {})
    b, c = baseline["summary"], candidate["summary"]
    base = {"schema_version": SCHEMA_VERSION, "policy": p, "baseline": b, "candidate": c,
            "baseline_name": baseline["name"], "candidate_name": candidate["name"],
            "dataset_version": candidate["dataset_version"], "regressions": [], "fixes": []}

    for key, gate in (("dataset_version", "dataset_mismatch"), ("evaluator_version", "evaluator_mismatch")):
        if baseline[key] != candidate[key]:
            return {**base, "decision": "error", "exit_code": EXIT_CODES["error"],
                    "reasons": [{"gate": gate, "cases": [],
                                 "detail": f"baseline and candidate used different {key.replace('_', ' ')}s"}]}

    if (b["error_rate"] or 0) > p["max_error_rate"]:  # provider/setup failure, not a quality signal
        first = next((r["reasons"][0] for r in baseline["results"] if r["status"] == "error"), "")
        return {**base, "decision": "error", "exit_code": EXIT_CODES["error"],
                "reasons": [{"gate": "baseline_errors", "cases": [],
                             "detail": f"baseline itself errored on {b['errors']}/{b['cases']} cases; fix setup first. First error: {first[:160]}"}]}

    bmap = {r["id"]: r for r in baseline["results"]}
    regressions = [r["id"] for r in candidate["results"]
                   if bmap.get(r["id"], {}).get("status") == "pass" and r["status"] != "pass"]
    fixes = [r["id"] for r in candidate["results"]
             if bmap.get(r["id"], {}).get("status") != "pass" and r["status"] == "pass"]
    reasons: list[dict] = []

    def fail(gate: str, detail: str, cases: list[str] | None = None):
        reasons.append({"gate": gate, "detail": detail, "cases": cases or []})

    if c["cases"] < p["min_cases"]:
        decision = "inconclusive"
        fail("min_cases", f"only {c['cases']} cases (< {p['min_cases']}); not enough evidence")
    else:
        drop = (b["pass_rate"] or 0) - (c["pass_rate"] or 0)
        if drop > p["max_pass_rate_drop"]:
            fail("pass_rate", f"pass rate fell {b['pass_rate']:.0%} → {c['pass_rate']:.0%} "
                 f"(max drop {p['max_pass_rate_drop']:.0%})", regressions)
        if (c["error_rate"] or 0) > p["max_error_rate"]:
            fail("error_rate", f"error rate {c['error_rate']:.0%} exceeds {p['max_error_rate']:.0%}",
                 [r["id"] for r in candidate["results"] if r["status"] == "error"])
        if len(regressions) > p["max_regressions"]:
            fail("regressions", f"{len(regressions)} case(s) passed in baseline but fail in candidate", regressions)
        if b["latency_p95_ms"] and c["latency_p95_ms"] is not None:
            inc = (c["latency_p95_ms"] / b["latency_p95_ms"] - 1) * 100
            if inc > p["max_p95_latency_increase_pct"]:
                fail("latency_p95", f"p95 latency +{inc:.0f}% ({b['latency_p95_ms']}ms → {c['latency_p95_ms']}ms)")
        skipped = None
        if b["total_cost_usd"] and c["total_cost_usd"] is not None:
            inc = (c["total_cost_usd"] / b["total_cost_usd"] - 1) * 100
            if inc > p["max_cost_increase_pct"]:
                fail("cost", f"cost +{inc:.0f}% (${b['total_cost_usd']:.6f} → ${c['total_cost_usd']:.6f})")
        else:
            skipped = "cost unavailable for one side; cost gate skipped"
        decision = "fail" if reasons else "pass"
        if skipped:
            base["notes"] = [skipped]

    return {**base, "decision": decision, "exit_code": EXIT_CODES[decision], "reasons": reasons,
            "regressions": regressions, "fixes": fixes}
