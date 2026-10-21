"""Measure EvalBeacon's per-case runner overhead with tracing off versus on.

  uv run python scripts/bench.py                       # in-process only (no network): the default
  uv run python scripts/bench.py --otlp                # additionally measure a real OTLP/gRPC exporter
  uv run python scripts/bench.py --cases 1000 --repeats 30 --json results/bench.json

Method
- The agent is a no-op, so the numbers isolate the runner + instrumentation cost, not model latency.
- Modes are INTERLEAVED within each repetition (the order alternates every repetition) so slow machine drift,
  thermal throttling and background load hit every mode equally. The earlier version ran all "off" runs and
  then all "on" runs, which attributes drift to tracing.
- "off": a no-op tracer. "on": the SDK tracer with a SimpleSpanProcessor and an in-memory exporter (in-process
  only). "otlp" (opt-in): SDK tracer with BatchSpanProcessor and the OTLP gRPC exporter to
  OTEL_EXPORTER_OTLP_ENDPOINT (default http://localhost:4317); it needs a reachable collector and includes
  network and serialisation cost, so it is reported separately and never mixed into "on".
- Warm-up runs are discarded. The overhead is the PAIRED difference (mode minus off) of each repetition;
  we report its median, spread and how many repetitions were positive. No significance test is performed.
- Not done: CPU pinning/affinity, process priority changes or frequency locking, so results include normal OS noise.
"""
import argparse
import gc
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

from src.evalbeacon import core  # noqa: E402
from src.evalbeacon.core import Case, Output, run_eval  # noqa: E402


def quantile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    pos = q * (len(s) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def summarize(xs: list[float]) -> dict:
    return {"median": statistics.median(xs), "mean": statistics.fmean(xs), "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs), "max": max(xs), "q1": quantile(xs, .25), "q3": quantile(xs, .75)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=int, default=500)
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--otlp", action="store_true", help="also measure a real OTLP gRPC exporter (needs a collector)")
    ap.add_argument("--json", help="write the full result to this path")
    a = ap.parse_args()

    cases = [Case(f"c{i}", {"q": i}, {"must_contain": ["ok"]}) for i in range(a.cases)]
    exporter = InMemorySpanExporter()
    provider_mem = TracerProvider()
    provider_mem.add_span_processor(SimpleSpanProcessor(exporter))
    tracers = {"off": trace.NoOpTracer(), "on": provider_mem.get_tracer("evalbeacon")}
    provider_otlp = None
    if a.otlp:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        provider_otlp = TracerProvider()
        provider_otlp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True, timeout=5)))
        tracers["otlp"] = provider_otlp.get_tracer("evalbeacon")
    modes = list(tracers)

    def run_once(mode: str) -> float:
        core.trace.get_tracer = lambda *_a, **_k: tracers[mode]  # swap the tracer the runner asks for
        exporter.clear()
        gc.collect()
        t = time.perf_counter()
        run_eval("bench", lambda _: Output("ok", 10, 5), cases, "v", model="gpt-4o-mini", workers=a.workers)
        return float((time.perf_counter() - t) / a.cases * 1000)  # ms per case

    for _ in range(a.warmup):
        for m in modes:
            run_once(m)
    samples: dict[str, list[float]] = {m: [] for m in modes}
    for rep in range(a.repeats):
        for m in (modes if rep % 2 == 0 else list(reversed(modes))):  # alternate order each repetition
            samples[m].append(run_once(m))
    if provider_otlp:
        provider_otlp.force_flush()
        provider_otlp.shutdown()

    result = {
        "context": {"python": platform.python_version(), "implementation": platform.python_implementation(),
                    "os": f"{platform.system()} {platform.release()}", "machine": platform.machine(),
                    "processor": platform.processor() or "unknown", "cpu_count": os.cpu_count(), "workers": a.workers,
                    "cases_per_run": a.cases, "repeats": a.repeats, "warmup_runs": a.warmup,
                    "interleaved": True, "cpu_pinning": False, "unit": "ms per case"},
        "modes": {m: summarize(v) for m, v in samples.items()},
        "paired_overhead_vs_off": {},
    }
    for m in modes[1:] if modes[0] == "off" else [x for x in modes if x != "off"]:
        diffs = [on - off for on, off in zip(samples[m], samples["off"], strict=True)]
        result["paired_overhead_vs_off"][m] = {**summarize(diffs), "positive_repetitions": f"{sum(d > 0 for d in diffs)}/{len(diffs)}"}

    c = result["context"]
    print(f"python {c['python']} | {c['os']} {c['machine']} | {c['cpu_count']} CPUs | {a.workers} workers | "
          f"{a.cases} cases x {a.repeats} repetitions (+{a.warmup} warm-up), interleaved")
    for m, s in result["modes"].items():
        print(f"  {m:<5} median {s['median']:.3f}  IQR [{s['q1']:.3f}, {s['q3']:.3f}]  min {s['min']:.3f}  max {s['max']:.3f}  stdev {s['stdev']:.3f}  ms/case")
    for m, s in result["paired_overhead_vs_off"].items():
        print(f"  overhead {m} - off: median {s['median']:+.3f} ms/case  IQR [{s['q1']:+.3f}, {s['q3']:+.3f}]  "
              f"range [{s['min']:+.3f}, {s['max']:+.3f}]  positive in {s['positive_repetitions']} repetitions")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
