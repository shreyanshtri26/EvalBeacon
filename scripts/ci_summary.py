"""Render results/report.json as a Markdown job summary (GitHub step summary), including trace IDs.

  python scripts/ci_summary.py results/report.json >> "$GITHUB_STEP_SUMMARY"
"""
import json
import sys
from pathlib import Path


def render(report: dict) -> str:
    g, cand = report["gate"], report["candidate"]
    traces = {r["id"]: r.get("trace_id") for r in cand["results"]}
    icon = {"pass": "✅", "fail": "❌", "error": "⚠️", "inconclusive": "❔"}.get(g["decision"], "")
    b, c = g["baseline"], g["candidate"]
    out = [f"### {icon} EvalBeacon gate: {g['decision'].upper()} (exit {g['exit_code']})",
           f"`{g['baseline_name']}` → `{g['candidate_name']}` on dataset `{g['dataset_version']}`", "",
           "| metric | baseline | candidate |", "|---|---|---|"]
    for k in ("pass_rate", "error_rate", "latency_p50_ms", "latency_p95_ms", "tokens_in", "tokens_out", "total_cost_usd"):
        out.append(f"| {k} | {b[k]} | {c[k]} |")
    if g["reasons"]:
        out += ["", "**Why:**"]
        for r in g["reasons"]:
            out.append(f"- **{r['gate']}**: {r['detail']}")
            for cid in r["cases"]:
                t = traces.get(cid)
                out.append(f"  - `{cid}` · trace `{t}`" if t else f"  - `{cid}` · no trace")
    out += ["", "**Policy:** " + ", ".join(f"`{k}={v}`" for k, v in g["policy"].items())]
    if g.get("notes"):
        out += ["", *[f"> {n}" for n in g["notes"]]]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # the summary contains emoji; Windows consoles default to cp1252
    if len(sys.argv) != 2:
        print("usage: ci_summary.py REPORT.json", file=sys.stderr)
        sys.exit(64)
    try:
        print(render(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))), end="")
    except (OSError, ValueError, KeyError) as e:
        print(f"### ⚠️ EvalBeacon summary unavailable\n\n`{type(e).__name__}: {e}`")  # never fail the CI job from the summary
