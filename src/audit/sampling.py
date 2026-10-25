"""Deterministic, seeded, stratified task selection. No global RNG state: order is a hash of (seed, task id)."""
from __future__ import annotations

import hashlib
from collections import defaultdict


def _rank(seed: str, task_id: str) -> str:
    return hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()


def stratified(public: dict[str, dict], per_stratum: int, seed: str, key: str = "repo") -> list[str]:
    """Pick up to `per_stratum` tasks from every stratum (default: repository); stable for a given seed."""
    strata: dict[str, list[str]] = defaultdict(list)
    for tid, t in public.items():
        strata[str(t[key])].append(tid)
    picked: list[str] = []
    for name in sorted(strata):
        picked += sorted(strata[name], key=lambda i: _rank(seed, i))[:per_stratum]
    return sorted(picked)


def proportional(public: dict[str, dict], n: int, seed: str, key: str = "repo") -> tuple[list[str], dict[str, float]]:
    """Proportional allocation with at least one task per stratum. Returns (ids, inclusion weight per id).

    The weight (stratum size / stratum sample size) lets later estimators stay unbiased for the whole benchmark."""
    strata: dict[str, list[str]] = defaultdict(list)
    for tid, t in public.items():
        strata[str(t[key])].append(tid)
    total = len(public)
    quota = {s: max(1, round(n * len(ids) / total)) for s, ids in strata.items()}
    ids: list[str] = []
    weights: dict[str, float] = {}
    for s in sorted(strata):
        chosen = sorted(strata[s], key=lambda i: _rank(seed, i))[:quota[s]]
        for i in chosen:
            weights[i] = len(strata[s]) / len(chosen)
        ids += chosen
    return sorted(ids), weights
