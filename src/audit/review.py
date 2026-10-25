"""Human-review support: blinded annotation packets, inter-annotator agreement, defect catalogue."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.audit import db, sampling, stats

PACKET_FIELDS = ("instance_id", "repo", "base_commit", "problem_statement")  # nothing about any model or patch


def annotation_sample(public: dict[str, dict], n: int, pilot: int, seed: str) -> dict:
    """Seeded proportional sample; the pilot is the first `pilot` ids by seeded hash rank (itself a stratified-free subset)."""
    ids, weights = sampling.proportional(public, n, seed)
    order = sorted(ids, key=lambda i: sampling._rank(seed + ":pilot", i))
    return {"seed": seed, "n": len(ids), "ids": ids, "pilot": sorted(order[:pilot]), "weights": weights}


def packets(ids: list[str], public: dict[str, dict]) -> list[dict]:
    return [{k: public[i][k] for k in PACKET_FIELDS} for i in ids]


def agreement(con: sqlite3.Connection, a: str, b: str) -> dict:
    """Cohen's kappa between two annotators on the (task, subject) items both labelled (latest label each)."""
    rows = con.execute(
        "SELECT task_id, subject, annotator, label, category FROM annotations WHERE id IN "
        "(SELECT MAX(id) FROM annotations WHERE annotator IN (?,?) GROUP BY task_id, subject, annotator)", (a, b)).fetchall()
    by: dict[tuple, dict[str, dict]] = {}
    for r in rows:
        by.setdefault((r["task_id"], r["subject"]), {})[r["annotator"]] = dict(r)
    both = {k: v for k, v in by.items() if a in v and b in v}
    if not both:
        return {"items": 0, "kappa_label": None, "kappa_category": None, "observed_agreement": None, "disagreements": []}
    la, lb = [v[a]["label"] for v in both.values()], [v[b]["label"] for v in both.values()]
    ca, cb = [v[a]["category"] for v in both.values()], [v[b]["category"] for v in both.values()]
    dis = [{"task_id": k[0], "subject": k[1], "a": v[a]["label"] + "/" + v[a]["category"],
            "b": v[b]["label"] + "/" + v[b]["category"]} for k, v in both.items()
           if (v[a]["label"], v[a]["category"]) != (v[b]["label"], v[b]["category"])]
    return {"items": len(both), "kappa_label": stats.cohen_kappa(la, lb), "kappa_category": stats.cohen_kappa(ca, cb),
            "observed_agreement": sum(x == y for x, y in zip(la, lb, strict=True)) / len(la), "disagreements": dis}


def catalogue(con: sqlite3.Connection) -> list[dict]:
    """Defect catalogue: one row per adjudicated task with the decision, its reason and the evidence reference."""
    return [{"task_id": r["task_id"], "subject": r["subject"], "status": r["status"], "disposition": r["disposition"],
             "reason": r["reason"], "evidence_ref": r["evidence_ref"]} for r in db.latest_adjudications(con)]


def defect_share(con: sqlite3.Connection, ids: list[str]) -> dict | None:
    """Share of `ids` with a verified task_defect adjudication, Wilson 95% CI (H3). None when nothing is adjudicated."""
    done = {r["task_id"]: r for r in db.latest_adjudications(con) if r["task_id"] in set(ids) and r["status"] != "suspect"}
    if not done:
        return None
    k = sum(r["disposition"] == "task_defect" and r["status"] == "verified" for r in done.values())
    rate, low, high = stats.wilson(k, len(done)) or (0.0, 0.0, 0.0)
    return {"adjudicated": len(done), "defects": k, "rate": rate, "low": low, "high": high}


def write_json(path: Path, obj: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return path
