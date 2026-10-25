"""REST API for the audit dashboard.

Two modes, reported by /overview: "local" (benchmark data present: task browser, evidence, review workflow) and
"reports-only" (a hosted deploy: only the committed aggregate reports). Hidden tests, reference patches and eval
scripts are never served. Review writes need AUDIT_WRITE=1 and are meant for a single trusted local user."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.audit import db, review, tasks

router = APIRouter(prefix="/api/v1/audit")
ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "audit"
UI = Path(__file__).resolve().parents[1] / "evalbeacon" / "ui" / "audit.html"
_TASK = re.compile(r"^[A-Za-z0-9_.\-]+__[A-Za-z0-9_.\-]+-\d+$")
_NAME = re.compile(r"^[a-z0-9][a-z0-9._\-]{0,120}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
MAX_LOG = 400_000


def _data() -> Path:
    return Path(os.environ.get("AUDIT_DATA", ROOT / "data" / "swebench"))


def _local() -> bool:
    return (_data() / "public.jsonl").exists()


def _con():
    return db.connect(os.environ.get("AUDIT_DB") or None)


def _write_enabled() -> bool:
    return os.environ.get("AUDIT_WRITE") == "1"


def alias(subject: str) -> str:
    """Blinded stand-in for a submission name (model identity must not influence a reviewer)."""
    return "" if not subject else "subject-" + hashlib.sha1(subject.encode()).hexdigest()[:6]


def _report(name: str) -> dict | None:
    p = REPORTS / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@router.get("/overview")
def overview():
    reports = sorted(p.stem for p in REPORTS.glob("*.json")) if REPORTS.exists() else []
    out: dict = {"mode": "local" if _local() else "reports-only", "reports": reports,
                 "write_enabled": _write_enabled(), "dataset": tasks.DATASET}
    if _local():
        pub = tasks.load_public(_data())
        out["tasks"] = len(pub)
        out["repo_shares"] = tasks.repo_shares(pub)
        con = _con()
        out["grader_runs"] = [dict(r) for r in con.execute(
            "SELECT subject, COUNT(*) AS runs, SUM(resolved) AS resolved FROM grader_runs GROUP BY subject")]
        out["annotations"] = con.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]
        out["adjudications"] = {s: n for s, n in con.execute(
            "SELECT status, COUNT(*) FROM (SELECT status FROM adjudications WHERE id IN "
            "(SELECT MAX(id) FROM adjudications GROUP BY task_id, subject)) GROUP BY status")}
    return out


@router.get("/reports/{name}")
def report(name: str):
    if not _NAME.match(name):
        raise HTTPException(400, "invalid report name")
    r = _report(name)
    if r is None:
        raise HTTPException(404, "report not found")
    return r


@router.get("/tasks")
def task_list(repo: str | None = None, q: str | None = None, limit: int = Query(100, ge=1, le=500),
              offset: int = Query(0, ge=0)):
    if not _local():
        raise HTTPException(404, "benchmark data is not present on this deployment (reports-only mode)")
    pub = tasks.load_public(_data())
    ids = [i for i, t in pub.items() if (not repo or t["repo"] == repo) and (not q or q.lower() in i.lower())]
    con = _con()
    last = {(r["task_id"], r["subject"]): r for r in con.execute(
        "SELECT task_id, subject, resolved, f2p_passed, f2p_total FROM grader_runs WHERE id IN "
        "(SELECT MAX(id) FROM grader_runs GROUP BY task_id, subject)")}
    rows = []
    for i in sorted(ids)[offset: offset + limit]:
        t = pub[i]
        runs = {s: ("resolved" if r["resolved"] else "unresolved") for (tid, s), r in last.items() if tid == i}
        rows.append({"id": i, "repo": t["repo"], "difficulty": t["difficulty"], "runs": runs})
    return {"total": len(ids), "rows": rows}


@router.get("/tasks/{task_id}")
def task_detail(task_id: str, as_: str | None = Query(None, alias="as"), reveal: bool = False):
    """Public task view, every grader run with its evidence hash, and the review trail.

    Blinding: a reviewer only sees their own annotations. `reveal=1` shows others only after they have labelled this
    task themselves; submission names are always replaced by aliases here."""
    if not _TASK.match(task_id):
        raise HTTPException(400, "invalid task id")
    if not _local():
        raise HTTPException(404, "benchmark data is not present on this deployment (reports-only mode)")
    pub = tasks.load_public(_data())
    if task_id not in pub:
        raise HTTPException(404, "unknown task")
    con = _con()
    runs = [dict(r) for r in con.execute(
        "SELECT id, run_id, subject, repeat, grader, resolved, patch_applied, infra_failure, f2p_passed, f2p_total, "
        "p2p_passed, p2p_total, failed_tests, reason, patch_sha256, log_sha256, image, image_id, network, memory, "
        "cpus, timeout_s, duration_s, timed_out FROM grader_runs WHERE task_id=? ORDER BY id", (task_id,))]
    for r in runs:
        r["failed_tests"] = json.loads(r["failed_tests"] or "[]")
        r["subject_alias"] = alias(r["subject"])
        if r["subject"] not in ("gold", "empty"):
            r.pop("subject")
    anns = [dict(r) for r in con.execute("SELECT * FROM annotations WHERE task_id=? ORDER BY id", (task_id,))]
    mine = [a for a in anns if a["annotator"] == as_] if as_ else []
    visible = anns if (reveal and mine) else mine
    adj = [dict(r) for r in con.execute("SELECT * FROM adjudications WHERE task_id=? ORDER BY id", (task_id,))]
    for r in adj + visible:
        if r["subject"] not in ("", "gold", "empty"):
            r["subject"] = alias(r["subject"])
    return {"task": pub[task_id], "runs": runs, "annotations": visible, "adjudications": adj}


@router.get("/evidence/{sha}")
def evidence(sha: str):
    if not _SHA.match(sha):
        raise HTTPException(400, "invalid evidence hash")
    try:
        text = db.get_evidence(sha)
    except FileNotFoundError:
        raise HTTPException(404, "evidence not found on this deployment") from None
    return {"sha256": sha, "bytes": len(text), "truncated": len(text) > MAX_LOG,
            "log": text if len(text) <= MAX_LOG else text[:MAX_LOG // 4] + "\n...[truncated]...\n" + text[-MAX_LOG * 3 // 4:]}


class AnnotationIn(BaseModel):
    task_id: str = Field(pattern=_TASK.pattern)
    annotator: str = Field(min_length=1, max_length=60)
    label: str
    category: str
    subject: str = ""
    confidence: float | None = Field(None, ge=0, le=1)
    notes: str = Field("", max_length=4000)
    evidence_ref: str | None = Field(None, pattern=_SHA.pattern)


class AdjudicationIn(BaseModel):
    task_id: str = Field(pattern=_TASK.pattern)
    adjudicator: str = Field(min_length=1, max_length=60)
    status: str
    disposition: str
    reason: str = Field(min_length=1, max_length=4000)
    subject: str = ""
    evidence_ref: str | None = Field(None, pattern=_SHA.pattern)


def _require_write() -> None:
    if not _write_enabled():
        raise HTTPException(403, "review writes are disabled on this deployment (set AUDIT_WRITE=1 locally)")


@router.post("/annotations", status_code=201)
def post_annotation(a: AnnotationIn):
    _require_write()
    try:
        return {"id": db.add_annotation(_con(), **a.model_dump())}
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@router.post("/adjudications", status_code=201)
def post_adjudication(a: AdjudicationIn):
    _require_write()
    try:
        return {"id": db.add_adjudication(_con(), **a.model_dump())}
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@router.get("/agreement")
def agreement(a: str, b: str):
    """Cohen's kappa between two annotators on the (task, subject) items both have labelled (latest label each)."""
    out = review.agreement(_con(), a, b)
    for d in out["disagreements"]:
        d["subject"] = alias(d["subject"])
    return out


pages = APIRouter()


@pages.get("/audit", include_in_schema=False)
def page():
    return FileResponse(UI)
