"""EvalBeacon CLI.

  evalbeacon demo [--candidate candidate-bad|candidate-good]   offline baseline vs candidate + gate
  evalbeacon live --provider openai|deepseek|gemini|custom     real model, needs a key in .env
  evalbeacon compare BASELINE.json CANDIDATE.json              gate two saved runs
  evalbeacon drift --golden RUN.json --current prod.jsonl     production-vs-golden drift signal
  evalbeacon replay [--record]                                 re-run with recorded (mocked) tool results
  evalbeacon history                                           list saved runs
  evalbeacon serve                                             dashboard at /ui

Exit codes: 0 pass/stable · 1 gate failed/drift · 2 setup or usage error · 3 inconclusive.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
from pathlib import Path

from src.evalbeacon import history
from src.evalbeacon.core import EXIT_CODES, SCHEMA_VERSION, compare, load_dataset, run_eval
from src.evalbeacon.redact import install_log_redaction
from src.evalbeacon.store import LocalPayloadStore

RESULTS = Path("results")


def _report(base: dict, cand: dict, policy: dict | None) -> dict:
    return {"schema_version": SCHEMA_VERSION, "baseline": base, "candidate": cand, "gate": compare(base, cand, policy)}


def _policy(path: str | None) -> dict | None:
    return json.loads(Path(path).read_text()) if path else None


def build_report(dataset: str, candidate: str, policy_path: str | None, model: str = "gpt-4o-mini",
                 store=None) -> dict:
    from src.evalbeacon.demo import AGENTS

    cases, version = load_dataset(dataset)
    policy = _policy(policy_path)
    base = run_eval("baseline", AGENTS["baseline"], cases, version, model=model, store=store)
    cand = run_eval(candidate, AGENTS[candidate], cases, version, model=model, store=store)
    return _report(base, cand, policy)


def build_live_report(dataset: str, provider: str, model: str | None, policy_path: str | None, *,
                      workers: int = 4, timeout: float = 30.0, judge_provider: str | None = None,
                      store=None, max_retries: int = 1, api_key: str | None = None, max_cases: int | None = None) -> dict:
    from src.evalbeacon.live import make_agent, make_judge

    cases, version = load_dataset(dataset)
    if max_cases:
        cases = cases[:max_cases]  # NB: the dataset version still names the full file; the report notes the cap
    policy = _policy(policy_path)
    judge = make_judge(judge_provider) if judge_provider else None
    runs = []
    for name, grounded in (("baseline", True), ("candidate-ungrounded", False)):
        agent, m = make_agent(provider, model, grounded=grounded, max_retries=max_retries, api_key=api_key)
        runs.append(run_eval(name, agent, cases, version, model=m, workers=workers, timeout_s=timeout,
                             judge=judge, store=store, revision="grounded" if grounded else "ungrounded"))
    return _report(runs[0], runs[1], policy)


def save(report: dict, out: Path = RESULTS, keep: int = 50) -> None:
    out.mkdir(exist_ok=True)
    (out / "baseline.json").write_text(json.dumps(report["baseline"], indent=2))
    (out / "candidate.json").write_text(json.dumps(report["candidate"], indent=2))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    for run in (report["baseline"], report["candidate"]):  # idempotent by run_id, pruned to `keep`
        history.save_run(run, out / "history", keep)


def _cases_with_traces(ids: list[str], run: dict | None) -> str:
    """case-id (trace abc123...) so a developer can open the failing case's trace directly."""
    traces = {r["id"]: r.get("trace_id") for r in (run or {}).get("results", [])}
    return ", ".join(f"{i} (trace {t[:12]}…)" if (t := traces.get(i)) else f"{i} (no trace)" for i in ids)


def print_gate(g: dict, candidate: dict | None = None) -> None:
    b, c = g["baseline"], g["candidate"]
    print(f"\n EvalBeacon · {g['baseline_name']} → {g['candidate_name']} · dataset {g['dataset_version']}")
    print(f" {'metric':<18}{'baseline':>12}{'candidate':>12}")
    for k in ("pass_rate", "error_rate", "latency_p50_ms", "latency_p95_ms", "tokens_in", "tokens_out",
              "total_cost_usd", "cost_per_success_usd"):
        print(f" {k:<18}{b[k]!s:>12}{c[k]!s:>12}")
    print(f"\n GATE: {g['decision'].upper()}")
    for r in g["reasons"]:
        print(f"  ✗ [{r['gate']}] {r['detail']}")
        if r["cases"]:
            print(f"      cases: {_cases_with_traces(r['cases'], candidate)}")
    for n in g.get("notes", []):
        print(f"  · {n}")


def _replay(record: bool, dataset: str) -> int:
    from src.evalbeacon.demo import kb_search, toolbox_baseline
    from src.evalbeacon.replay import Cassette, Recorder, Replayer

    cassette_path = RESULTS / "cassette.json"
    cases, version = load_dataset(dataset)
    if record:
        cassette = Cassette()

        def mk():
            return Recorder({"kb_search": kb_search}, cassette)  # fresh call log per case
        label = "recorded"
    else:
        cassette = Cassette.load(cassette_path)

        def mk():
            return Replayer(cassette)
        label = "replayed (no live tools)"
    run = run_eval(label, lambda inp: toolbox_baseline(inp, mk()), cases, version, workers=1)
    if record:
        cassette.save(cassette_path)
        print(f" recorded {len(cassette.entries)} tool results → {cassette_path}")
    s = run["summary"]
    print(f" {label}: {s['passed']}/{s['cases']} passed, {s['errors']} errors")
    for f in run["failure_summary"]:
        print(f"  ✗ {f['id']}: {f['reason']}")
    return 0 if run["status"] == "complete" and s["passed"] == s["cases"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evalbeacon", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--dataset", default="datasets/support.jsonl")
    d.add_argument("--candidate", default="candidate-bad", choices=["candidate-bad", "candidate-good", "candidate-flaky"])
    d.add_argument("--policy", default="configs/gate.json")
    lv = sub.add_parser("live", help="baseline vs candidate against a real provider (needs .env key)")
    lv.add_argument("--provider", required=True)
    lv.add_argument("--model")
    lv.add_argument("--dataset", default="datasets/support.jsonl")
    lv.add_argument("--policy", default="configs/gate.json")
    lv.add_argument("--workers", type=int, default=4)
    lv.add_argument("--timeout", type=float, default=30.0)
    lv.add_argument("--max-retries", type=int, default=1, help="SDK transport retries per request (0 = strict one request per case)")
    lv.add_argument("--judge-provider", help="enable the optional LLM judge for cases that define one")
    c = sub.add_parser("compare")
    c.add_argument("baseline")
    c.add_argument("candidate")
    c.add_argument("--policy")
    dr = sub.add_parser("drift")
    dr.add_argument("--golden", default="results/baseline.json")
    dr.add_argument("--current", required=True, help="JSONL of {id, output} from production")
    dr.add_argument("--threshold", type=float, default=0.6)
    dr.add_argument("--min-cases", type=int, default=5)
    dr.add_argument("--max-drifted-pct", type=float, default=20.0)
    rp = sub.add_parser("replay")
    rp.add_argument("--record", action="store_true")
    rp.add_argument("--dataset", default="datasets/support.jsonl")
    sub.add_parser("history")
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8000)
    for p in (d, lv, c):
        p.add_argument("--keep", type=int, default=50, help="runs kept in results/history (retention)")
        p.add_argument("--store", action="store_true", help="persist redacted payloads to results/payloads")
    a = ap.parse_args(argv)
    install_log_redaction()  # no secret may reach a log line, whatever the provider error says
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
        sys.stdout.reconfigure(encoding="utf-8")
    if a.cmd not in ("serve", "history"):
        _init_tracing()
    store = LocalPayloadStore(RESULTS / "payloads") if getattr(a, "store", False) else None

    try:
        if a.cmd == "serve":
            import uvicorn
            uvicorn.run("src.api.main:app", host="127.0.0.1", port=a.port)
            return 0
        if a.cmd == "history":
            rows = history.list_runs(RESULTS / "history")
            for r in rows:
                print(f" {r['run_id']}  {r['name']:<22}{r['status']:<10}{r['cases']:>4} cases  pass {r['pass_rate']}")
            print(f" {len(rows)} run(s)")
            return 0
        if a.cmd == "replay":
            return _replay(a.record, a.dataset)
        if a.cmd == "drift":
            from src.evalbeacon.drift import detect, load_current
            res = detect(json.loads(Path(a.golden).read_text()), load_current(a.current), threshold=a.threshold,
                         min_cases=a.min_cases, max_drifted_pct=a.max_drifted_pct)
            print(f"\n DRIFT: {res['decision'].upper()} · {res['compared']} compared · mean similarity "
                  f"{res['mean_similarity']} ({res['method']}) · {res['drifted_pct']}% below {res['threshold']}")
            if res["drifted"]:
                print(f"  drifted: {', '.join(res['drifted'])}")
            return int(res["exit_code"])
        if a.cmd == "demo":
            report = build_report(a.dataset, a.candidate, a.policy, store=store)
        elif a.cmd == "live":
            report = build_live_report(a.dataset, a.provider, a.model, a.policy, workers=a.workers,
                                       timeout=a.timeout, judge_provider=a.judge_provider, store=store,
                                       max_retries=a.max_retries)
        else:
            base, cand = (json.loads(Path(p).read_text()) for p in (a.baseline, a.candidate))
            report = _report(base, cand, _policy(a.policy))
        save(report, keep=a.keep)
    except (OSError, ValueError, KeyError) as e:
        print(f"evalbeacon: {e}", file=sys.stderr)
        return EXIT_CODES["error"]
    print_gate(report["gate"], report["candidate"])
    return int(report["gate"]["exit_code"])


def _init_tracing() -> None:
    """SDK provider so every case gets a real trace_id; EVALBEACON_TRACE=console also prints the spans."""
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "evalbeacon")}))
    mode = os.getenv("EVALBEACON_TRACE")
    if mode == "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    elif mode == "otlp":  # OTEL_EXPORTER_OTLP_ENDPOINT (default http://localhost:4317), gRPC
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=endpoint.startswith("http://"),
                                                                        timeout=5)))
        atexit.register(provider.shutdown)  # flush queued spans before the CLI exits
    trace.set_tracer_provider(provider)


if __name__ == "__main__":
    sys.exit(main())
