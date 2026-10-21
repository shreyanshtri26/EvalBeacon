"""Production-vs-golden drift signal: per-case output similarity against a golden run.

Lexical similarity by default (deterministic); pass embed_fn for semantic similarity.
This is a signal for review, not proof of a quality change.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.evalbeacon.redact import redact
from src.evalbeacon.similarity import EmbedFn, similarity

EXIT = {"stable": 0, "drift": 1, "error": 2, "inconclusive": 3}


def load_current(path: str | Path) -> dict[str, str]:
    out = {}
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            out[d["id"]] = d["output"]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise ValueError(f"{path}:{n}: invalid record ({e!r})") from e
    return out


def detect(golden: dict, current: dict[str, str], *, threshold: float = 0.6, min_cases: int = 5,
           max_drifted_pct: float = 20.0, embed_fn: EmbedFn | None = None) -> dict:
    gold = {r["id"]: r.get("output") for r in golden["results"] if r["status"] == "pass" and r.get("output")}
    shared = [i for i in gold if i in current]
    sims = {i: similarity(gold[i], redact(current[i]), embed_fn) for i in shared}  # golden outputs are stored redacted
    drifted = sorted(i for i, s in sims.items() if s < threshold)
    pct = 100 * len(drifted) / len(sims) if sims else 0.0
    decision = ("inconclusive" if len(sims) < min_cases else "drift" if pct > max_drifted_pct else "stable")
    return {"decision": decision, "exit_code": EXIT[decision], "method": "embedding" if embed_fn else "lexical-cosine",
            "threshold": threshold, "compared": len(sims), "missing_in_current": sorted(set(gold) - set(current)),
            "mean_similarity": round(sum(sims.values()) / len(sims), 4) if sims else None,
            "drifted_pct": round(pct, 1), "drifted": drifted, "similarities": sims,
            "golden_name": golden["name"], "dataset_version": golden["dataset_version"]}
