"""Bridge between SWE-bench grading and EvalBeacon's run/compare core: a grading pass is an EvalBeacon run."""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path

from src.audit import db, grader, sandbox
from src.evalbeacon.core import Case, Output, load_dataset, run_eval

PatchFor = Callable[[str], str | None]
RESOLVED_TOKEN = "resolved=true"


def write_cases(path: Path, ids: list[str], public: dict[str, dict], subject: str, *, expect_resolved: bool = True,
                extra_meta: dict | None = None) -> tuple[list[Case], str]:
    """Write an EvalBeacon dataset (JSONL, content-hashed) with one case per task; return (cases, version)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exp = {"must_contain": [RESOLVED_TOKEN]} if expect_resolved else {"must_not_contain": [RESOLVED_TOKEN]}
    lines = [json.dumps({"id": i, "input": {"instance_id": i, "subject": subject}, "expected": exp,
                         "metadata": {"repo": public[i]["repo"], "difficulty": public[i]["difficulty"],
                                      **(extra_meta or {})}}) for i in ids]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return load_dataset(path)


def grading_agent(subject: str, patch_for: PatchFor, public: dict[str, dict], restricted: dict[str, dict],
                  con: sqlite3.Connection, run_id: str, limits: sandbox.Limits = sandbox.Limits(),
                  repeat: int = 0) -> Callable[[dict], Output]:
    """An EvalBeacon 'agent' whose output is the grader verdict for a stored patch. Evidence is persisted."""
    def agent(inp: dict) -> Output:
        iid = inp["instance_id"]
        patch = patch_for(iid)
        if patch is None:
            raise LookupError(f"no patch for {iid} in subject {subject!r}")
        t, r = public[iid], restricted[iid]
        sr = sandbox.run_patch(t["image"], r["eval_script"], patch, limits)
        if sr.error:
            raise RuntimeError(sr.error)
        v = grader.grade(t, r, sr.log, patch)
        log_sha = db.put_evidence(sr.log)
        db.record_run(con, run_id=run_id, task_id=iid, subject=subject, repeat=repeat, grader=v.grader,
                      resolved=int(v.resolved), patch_applied=int(v.patch_applied), infra_failure=int(v.infra_failure),
                      f2p_passed=v.f2p_passed, f2p_total=v.f2p_total, p2p_passed=v.p2p_passed, p2p_total=v.p2p_total,
                      failed_tests=v.failed_tests, reason=v.reason, patch_sha256=sr.patch_sha256,
                      eval_script_sha256=sr.eval_script_sha256, log_sha256=log_sha, image=sr.image,
                      image_id=sr.image_id, docker_server=sr.docker_server, network=limits.network,
                      memory=limits.memory, cpus=limits.cpus, timeout_s=limits.timeout_s,
                      duration_s=round(sr.duration_s, 1), timed_out=int(sr.timed_out), error=None)
        return Output(text=f"resolved={str(v.resolved).lower()} f2p={v.f2p_passed}/{v.f2p_total} "
                           f"p2p={v.p2p_passed}/{v.p2p_total} {v.reason}".strip())
    return agent


def run_grading(name: str, cases: list[Case], version: str, agent: Callable[[dict], Output], *,
                timeout_s: float = 2400.0, workers: int = 1, revision: str = "unversioned") -> dict:
    """One EvalBeacon run over the cases; a sandbox crash becomes an 'error' result, never an exception."""
    return run_eval(name, agent, cases, version, model=grader.grader_version(), timeout_s=timeout_s,
                    workers=workers, revision=revision)
