"""Reference-patch and empty-patch controls. A harness that fails its own controls cannot be trusted for anything else."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from src.audit import runner, sandbox
from src.evalbeacon import history

CASES_DIR = Path("data/audit/cases")


def run_controls(ids: list[str], public: dict[str, dict], restricted: dict[str, dict], con: sqlite3.Connection, *,
                 limits: sandbox.Limits = sandbox.Limits(), workers: int = 1, label: str = "controls",
                 save_history: bool = True) -> dict:
    """Run gold (must resolve) and empty (must not resolve) for every id; report each mismatch explicitly."""
    out: dict = {"label": label, "tasks": ids, "limits": limits.__dict__, "controls": {}}
    for subject, expect in (("gold", True), ("empty", False)):
        def patch_for(i: str, subject: str = subject) -> str:
            return restricted[i]["patch"] if subject == "gold" else ""
        cases, version = runner.write_cases(CASES_DIR / f"{label}-{subject}.jsonl", ids, public, subject,
                                            expect_resolved=expect)
        run = runner.run_grading(f"{label}-{subject}", cases, version,
                                 runner.grading_agent(subject, patch_for, public, restricted, con,
                                                      f"{label}-{subject}", limits), workers=workers)
        if save_history:
            history.save_run(run, Path("results/audit-history"))
        bad = [r for r in run["results"] if r["status"] != "pass"]
        out["controls"][subject] = {
            "run_id": run["run_id"], "dataset_version": version, "expected_resolved": expect,
            "cases": len(run["results"]), "as_expected": len(run["results"]) - len(bad),
            "mismatches": [{"id": r["id"], "status": r["status"], "reasons": r["reasons"][:2]} for r in bad]}
    out["ok"] = all(c["as_expected"] == c["cases"] for c in out["controls"].values())
    return out
