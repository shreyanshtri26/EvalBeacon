"""Experiments over public submissions: comparability (Q6), grader reproducibility (Q1/Q5), leakage (Q4), sensitivity (Q7)."""
from __future__ import annotations

import random
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.audit import runner, sandbox, stats, submissions
from src.audit.submissions import Entry
from src.evalbeacon.core import Output, compare

# Protocol fields the matrix checks. "unstated" is a first-class value: absence in metadata.yaml is not "no".
MATRIX_FIELDS = ("agent", "model", "attempts", "reasoning_effort", "reported_cost_usd", "checked",
                 "trajs_available", "logs_available")


def comparability_matrix(split: str = "verified", workers: int = 8) -> dict:
    names = submissions.list_entries(split)
    with ThreadPoolExecutor(workers) as ex:
        entries = list(ex.map(lambda n: submissions.load_entry(n, split), names))
    rows = [e.comparability() for e in entries]
    stated = {f: sum(r[f] not in (None, False, "") for r in rows) for f in MATRIX_FIELDS}
    return {"split": split, "entries": len(rows), "stated": stated,
            "with_original_verdicts": sum(r["verdict_count"] > 0 for r in rows),
            "attempts_distribution": dict(Counter(str(r["attempts"]) for r in rows)),
            "agents": dict(Counter(str(r["agent"]) for r in rows).most_common(12)), "rows": rows}


def regrade(entry: Entry, ids: list[str], public: dict[str, dict], restricted: dict[str, dict],
            con: sqlite3.Connection, *, limits: sandbox.Limits = sandbox.Limits(), workers: int = 1,
            cases_dir: Path = Path("data/audit/cases")) -> dict:
    """Re-grade a submission's recorded patches in our sandbox and compare with the verdicts it originally reported.

    Disagreement is NOT a finding about truth: it can be an original false accept, an environment difference
    (network policy, image build date), or flakiness. It identifies tasks to examine."""
    trajs = {i: submissions.fetch_traj(entry, i) for i in ids}
    usable = [i for i in ids if trajs[i] is not None and i in entry.verdicts]
    skipped = {i: ("no trajectory" if trajs[i] is None else "no original verdict") for i in ids if i not in usable}
    if not usable:
        return {"entry": entry.name, "usable": 0, "skipped": skipped}

    def patch_for(i: str) -> str | None:
        t = trajs[i]
        return (t.patch or "") if t else None

    subject = entry.name
    cases, version = runner.write_cases(cases_dir / f"regrade-{entry.name}.jsonl", usable, public, subject,
                                        extra_meta={"entry": entry.name})

    def original(inp: dict) -> Output:
        return Output(text=f"resolved={str(entry.verdicts[inp['instance_id']]).lower()} (recorded verdict)")

    base = runner.run_grading(f"original:{entry.name}", cases, version, original, workers=4)
    cand = runner.run_grading(f"reproduced:{entry.name}", cases, version,
                              runner.grading_agent(subject, patch_for, public, restricted, con,
                                                   f"regrade-{entry.name}", limits), workers=workers)
    gate = compare(base, cand, {"min_cases": 1, "max_regressions": len(usable), "max_pass_rate_drop": 1.0,
                                "max_error_rate": 1.0, "max_p95_latency_increase_pct": 1e9,
                                "max_cost_increase_pct": 1e9})
    errored = {r["id"] for r in cand["results"] if r["status"] == "error"}
    orig_only = [i for i in gate["regressions"] if i not in errored]  # original resolved, reproduction did not
    ours_only = [i for i in gate["fixes"] if i not in errored]
    both = len(usable) - len(errored) - len(orig_only) - len(ours_only)
    return {"entry": entry.name, "usable": len(usable), "skipped": skipped, "errors": sorted(errored),
            "agree": both, "original_resolved_not_reproduced": orig_only, "reproduced_not_originally_resolved": ours_only,
            "mcnemar_p": stats.mcnemar_exact(len(ours_only), len(orig_only)), "limits": limits.__dict__,
            "baseline_run": base["run_id"], "candidate_run": cand["run_id"], "dataset_version": version}


def leak_summary(entry: Entry, ids: list[str], restricted: dict[str, dict], workers: int = 8) -> dict:
    """Scan trajectories for history probing and gold-patch exposure. Observable evidence only."""
    def one(i: str) -> tuple[str, dict | None]:
        t = submissions.fetch_traj(entry, i)
        return i, (submissions.leak_scan(t, restricted[i]["patch"]) if t else None)

    with ThreadPoolExecutor(workers) as ex:
        scans = dict(ex.map(one, ids))
    scanned = {i: s for i, s in scans.items() if s is not None}
    levels = Counter(s["level"] for s in scanned.values())
    flagged = {i: s for i, s in scanned.items() if s["level"] in ("gold_overlap", "broad_history")}
    n = len(scanned)
    return {"entry": entry.name, "requested": len(ids), "scanned": n, "no_trajectory": len(ids) - n,
            "levels": dict(levels),
            "gold_overlap_rate": stats.wilson(levels["gold_overlap"], n),
            "broad_history_rate": stats.wilson(levels["broad_history"] + levels["gold_overlap"], n),
            "flagged": {i: {"level": s["level"], "gold_overlap": s["gold_overlap"][:3],
                            "broad_history": s["broad_history"][:3]} for i, s in sorted(flagged.items())}}


def leak_study(entries: list[Entry], ids: list[str], restricted: dict[str, dict], *, workers: int = 8,
               boundary: str = "20250911") -> dict:
    """Scan every (entry, task) trajectory once; compare entries dated before vs from the image-fix `boundary`.

    Uncertainty is task-clustered. Entries differ in model as well as date, so the contrast is descriptive, not causal.
    Unparseable or missing trajectories are counted, never assumed clean."""
    jobs = [(e, i) for e in entries for i in ids]

    def one(job: tuple[Entry, str]) -> tuple[str, str, str, str | None]:
        e, i = job
        try:
            t = submissions.fetch_traj(e, i, keep=False)
        except ValueError:
            return e.name, i, "unparsed", None
        if t is None:
            return e.name, i, "missing", None
        return e.name, i, "ok", submissions.leak_scan(t, restricted[i]["patch"])["level"]

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, jobs))
    out: dict = {"boundary": boundary, "entries": [e.name for e in entries], "tasks": len(ids), "groups": {}}
    for label, pick in (("before", lambda n: n[:8] < boundary), ("from", lambda n: n[:8] >= boundary)):
        rows = [r for r in res if pick(r[0])]
        ok = [r for r in rows if r[2] == "ok"]
        g: dict = {"entries": len({r[0] for r in rows}), "trajectories": len(rows), "scanned": len(ok),
                   "missing": sum(r[2] == "missing" for r in rows), "unparsed": sum(r[2] == "unparsed" for r in rows),
                   "levels": dict(Counter(r[3] for r in ok))}
        for key, levels in (("gold_overlap", {"gold_overlap"}), ("broad_history_or_worse", {"broad_history", "gold_overlap"}),
                            ("any_history_command", {"history_commands", "broad_history", "gold_overlap"})):
            per_task: dict[str, list[int]] = {}
            for _, i, _, lvl in ok:
                k, n = per_task.setdefault(i, [0, 0])
                per_task[i] = [k + (lvl in levels), n + 1]
            g[key] = stats.cluster_ratio_ci([(k, n) for k, n in per_task.values()], seed=1)
        g["flagged"] = sorted({f"{r[0]}::{r[1]}" for r in ok if r[3] == "gold_overlap"})
        out["groups"][label] = g
    return out


def outcome_table(entries: list[Entry]) -> dict[str, dict[str, int]]:
    return {e.name: {i: int(v) for i, v in e.verdicts.items()} for e in entries if e.verdicts}


def null_rank_change(outcomes: dict[str, dict[str, int]], k: int, *, reps: int = 1000, seed: int = 0) -> dict | None:
    """Reference distribution: how much does the ranking move if k RANDOM tasks are removed?

    An adjudicated-defect exclusion only means something relative to this noise floor."""
    agents = sorted(outcomes)
    if len(agents) < 2:
        return None
    common = sorted(set.intersection(*(set(o) for o in outcomes.values())))
    if k >= len(common):
        return None
    full = {a: sum(outcomes[a][t] for t in common) / len(common) for a in agents}
    rng = random.Random(seed)
    taus = []
    for _ in range(reps):
        drop = set(rng.sample(common, k))
        kept = [t for t in common if t not in drop]
        taus.append(stats.kendall_tau(full, {a: sum(outcomes[a][t] for t in kept) / len(kept) for a in agents}))
    ok = sorted(t for t in taus if t is not None)
    return {"k": k, "agents": len(agents), "tasks": len(common), "reps": len(ok), "tau_mean": sum(ok) / len(ok),
            "tau_p05": ok[int(0.05 * len(ok))], "tau_median": ok[len(ok) // 2], "tau_min": ok[0]}
