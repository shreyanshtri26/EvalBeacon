"""End-to-end check of the docker-compose observability stack. Exits 0 only if every check passes.

  docker compose up -d --build
  uv run python scripts/verify_stack.py

It runs a real evaluation from the CLI with OTLP export, then looks each case's stored trace_id up in Jaeger,
and checks Prometheus targets and rules, Loki, Alertmanager, the collector and Grafana datasources.
Service URLs can be overridden with the *_URL environment variables below.
"""
import base64
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URLS = {k: os.getenv(f"{k}_URL", v) for k, v in {
    "JAEGER": "http://localhost:16686", "PROMETHEUS": "http://localhost:9090", "LOKI": "http://localhost:3100",
    "ALERTMANAGER": "http://localhost:9093", "COLLECTOR_HEALTH": "http://localhost:13133",
    "GRAFANA": "http://localhost:3000", "APP": "http://localhost:8000"}.items()}
results: list[tuple[str, bool, str]] = []


def get(url: str, auth: str | None = None, timeout: int = 10) -> str:
    req = urllib.request.Request(url)
    if auth:
        req.add_header("Authorization", "Basic " + base64.b64encode(auth.encode()).decode())
    return urllib.request.urlopen(req, timeout=timeout).read().decode()


def check(name: str, fn) -> None:
    try:
        detail = fn()
        results.append((name, True, str(detail or "")))
    except Exception as e:  # noqa: BLE001 — a failed check is a reported result, not a crash
        results.append((name, False, f"{type(e).__name__}: {str(e)[:140]}"))


def retry(fn, seconds=45):
    end = time.time() + seconds
    while True:
        try:
            return fn()
        except Exception:  # noqa: BLE001
            if time.time() > end:
                raise
            time.sleep(3)


def run_eval_with_otlp() -> dict:
    env = {**os.environ, "EVALBEACON_TRACE": "otlp", "OTEL_SERVICE_NAME": "evalbeacon-verify",
           "OTEL_EXPORTER_OTLP_ENDPOINT": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")}
    p = subprocess.run([sys.executable, "-m", "src.evalbeacon.cli", "demo", "--candidate", "candidate-flaky"],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    if p.returncode != 1:  # the flaky candidate must be blocked by the gate
        raise RuntimeError(f"expected gate exit code 1, got {p.returncode}: {p.stderr[-200:]}")
    return json.loads((ROOT / "results/candidate.json").read_text())


def trace_check(cand: dict) -> str:
    ok = next(r for r in cand["results"] if r["status"] == "pass")
    bad = next(r for r in cand["results"] if r["status"] == "error")
    out = []
    for label, r in (("pass", ok), ("error", bad)):
        spans = retry(lambda r=r: json.loads(get(f"{URLS['JAEGER']}/api/traces/{r['trace_id']}"))["data"][0]["spans"])
        tags = {t["key"]: t["value"] for t in spans[0]["tags"]}
        assert tags.get("evalbeacon.case_id") == r["id"], f"{label}: wrong case in trace"
        assert tags.get("evalbeacon.status") == r["status"], f"{label}: status mismatch"
        if label == "error":
            assert tags.get("otel.status_code") == "ERROR", "error case span is not marked ERROR in Jaeger"
        out.append(f"{label}:{r['id']}")
    return "found " + ", ".join(out)


cand: dict = {}
check("evaluation runs and gate blocks flaky candidate", lambda: cand.update(run_eval_with_otlp()) or f"{cand['summary']['passed']}/{cand['summary']['cases']} passed")
check("case spans reach Jaeger with matching trace_id/status", lambda: trace_check(cand))
check("app health", lambda: get(f"{URLS['APP']}/health"))
check("collector health", lambda: retry(lambda: get(f"{URLS['COLLECTOR_HEALTH']}/")))
def targets_up() -> dict:
    def once():
        t = {x["labels"]["job"]: x["health"] for x in json.loads(get(f"{URLS['PROMETHEUS']}/api/v1/targets"))["data"]["activeTargets"]}
        assert {"llm-app", "otel-collector"} <= set(t) and all(v == "up" for v in t.values()), t  # first scrape may take ~15s
        return t
    return retry(once, seconds=60)


check("prometheus scrapes llm-app and otel-collector", targets_up)
check("prometheus loaded alert rules", lambda: sum(len(g["rules"]) for g in json.loads(get(f"{URLS['PROMETHEUS']}/api/v1/rules"))["data"]["groups"]))
check("metrics from collector exporter present", lambda: retry(lambda: int(float(json.loads(get(
    f"{URLS['PROMETHEUS']}/api/v1/query?query=" + urllib.parse.quote('count({__name__=~"llm_observability.*"})')))["data"]["result"][0]["value"][1]))))
check("loki ready", lambda: retry(lambda: get(f"{URLS['LOKI']}/ready").strip()))
def loki_has_logs() -> int:
    """A freshly started app has logged nothing, so first make it log: a chat request that cannot succeed
    (no key in a clean CI runner, or a rejected key locally) emits a WARNING/ERROR that must reach Loki."""
    try:
        req = urllib.request.Request(f"{URLS['APP']}/api/v1/chat", data=json.dumps({"message": "log probe", "provider": "deepseek"}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=20)
    except Exception:  # noqa: BLE001 — 400/502 is the expected outcome; only the log line matters
        pass

    def once():
        q = urllib.parse.urlencode({"query": '{service_name=~".+"}', "limit": 1})
        streams = json.loads(get(f"{URLS['LOKI']}/loki/api/v1/query_range?{q}"))["data"]["result"]
        assert streams, "no log streams yet"
        return len(streams)
    return retry(once)


check("loki received logs from the app", loki_has_logs)
check("alertmanager ready", lambda: get(f"{URLS['ALERTMANAGER']}/-/ready").strip())
check("grafana datasources provisioned", lambda: sorted(d["name"] for d in json.loads(get(f"{URLS['GRAFANA']}/api/datasources", auth="admin:admin"))))

width = max(len(n) for n, _, _ in results)
for name, ok, detail in results:
    print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
failed = [n for n, ok, _ in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
