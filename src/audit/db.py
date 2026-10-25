"""SQLite store for grader runs and the human review trail. Evidence is append-only; derived tables are views."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

AUDIT_DIR = Path("data/audit")
RUBRIC_VERSION = "0"

# Rubric categories (v0); the preregistration freezes this list before the main annotation.
CATEGORIES = ("underspecified", "contradiction", "missing_acceptance", "overly_restrictive",
              "incorrect_test", "environment_dependent", "solution_leak", "other", "none")
LABELS = ("defect", "ok", "unsure")
STATUSES = ("verified", "suspect", "inconclusive")
DISPOSITIONS = ("false_accept", "false_reject", "correct_accept", "correct_reject", "task_defect", "undetermined")

SCHEMA = """
CREATE TABLE IF NOT EXISTS grader_runs (
  id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, task_id TEXT NOT NULL, subject TEXT NOT NULL,
  repeat INTEGER NOT NULL DEFAULT 0, grader TEXT NOT NULL, resolved INTEGER, patch_applied INTEGER,
  infra_failure INTEGER, f2p_passed INTEGER, f2p_total INTEGER, p2p_passed INTEGER, p2p_total INTEGER,
  failed_tests TEXT, reason TEXT, patch_sha256 TEXT, eval_script_sha256 TEXT, log_sha256 TEXT, image TEXT,
  image_id TEXT, docker_server TEXT, network TEXT, memory TEXT, cpus TEXT, timeout_s REAL, duration_s REAL,
  timed_out INTEGER, error TEXT, created_at REAL);
CREATE INDEX IF NOT EXISTS ix_runs_task ON grader_runs(task_id, subject);
CREATE TABLE IF NOT EXISTS annotations (
  id INTEGER PRIMARY KEY, task_id TEXT NOT NULL, subject TEXT NOT NULL DEFAULT '', annotator TEXT NOT NULL,
  rubric_version TEXT NOT NULL, label TEXT NOT NULL, category TEXT NOT NULL, confidence REAL, notes TEXT,
  evidence_ref TEXT, blind INTEGER NOT NULL DEFAULT 1, created_at REAL);
CREATE TABLE IF NOT EXISTS adjudications (
  id INTEGER PRIMARY KEY, task_id TEXT NOT NULL, subject TEXT NOT NULL DEFAULT '', status TEXT NOT NULL,
  disposition TEXT NOT NULL, adjudicator TEXT NOT NULL, reason TEXT NOT NULL, evidence_ref TEXT,
  created_at REAL);
CREATE INDEX IF NOT EXISTS ix_adj_task ON adjudications(task_id, subject);
"""


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    p = Path(path or os.environ.get("AUDIT_DB") or AUDIT_DIR / "audit.db")
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p, timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def put_evidence(log: str, root: Path | None = None) -> str:
    """Store a log content-addressed and immutable; return its sha256."""
    sha = hashlib.sha256(log.encode("utf-8")).hexdigest()
    f = (root or AUDIT_DIR / "evidence") / f"{sha}.log.gz"
    f.parent.mkdir(parents=True, exist_ok=True)
    if not f.exists():
        f.write_bytes(gzip.compress(log.encode("utf-8"), 6, mtime=0))
    return sha


def get_evidence(sha: str, root: Path | None = None) -> str:
    return gzip.decompress(((root or AUDIT_DIR / "evidence") / f"{sha}.log.gz").read_bytes()).decode("utf-8")


def record_run(con: sqlite3.Connection, **v) -> int:
    v.setdefault("repeat", 0)
    v["failed_tests"] = json.dumps(v.get("failed_tests", []))
    v["created_at"] = time.time()
    cols = ", ".join(v)
    cur = con.execute(f"INSERT INTO grader_runs ({cols}) VALUES ({', '.join('?' * len(v))})", list(v.values()))
    con.commit()
    return int(cur.lastrowid or 0)


def add_annotation(con: sqlite3.Connection, *, task_id: str, annotator: str, label: str, category: str,
                   subject: str = "", confidence: float | None = None, notes: str = "",
                   evidence_ref: str | None = None, blind: bool = True) -> int:
    if label not in LABELS or category not in CATEGORIES:
        raise ValueError(f"label must be in {LABELS} and category in {CATEGORIES}")
    if label == "defect" and category == "none":
        raise ValueError("a 'defect' label needs a category other than 'none'")
    cur = con.execute("INSERT INTO annotations (task_id, subject, annotator, rubric_version, label, category, "
                      "confidence, notes, evidence_ref, blind, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      (task_id, subject, annotator, RUBRIC_VERSION, label, category, confidence, notes,
                       evidence_ref, int(blind), time.time()))
    con.commit()
    return int(cur.lastrowid or 0)


def add_adjudication(con: sqlite3.Connection, *, task_id: str, adjudicator: str, status: str, disposition: str,
                     reason: str, subject: str = "", evidence_ref: str | None = None) -> int:
    if status not in STATUSES or disposition not in DISPOSITIONS:
        raise ValueError(f"status must be in {STATUSES} and disposition in {DISPOSITIONS}")
    if not reason.strip():
        raise ValueError("an adjudication needs a written reason")
    if status == "verified" and not evidence_ref:
        raise ValueError("'verified' requires an evidence_ref (log hash or run id)")
    cur = con.execute("INSERT INTO adjudications (task_id, subject, status, disposition, adjudicator, reason, "
                      "evidence_ref, created_at) VALUES (?,?,?,?,?,?,?,?)",
                      (task_id, subject, status, disposition, adjudicator, reason, evidence_ref, time.time()))
    con.commit()
    return int(cur.lastrowid or 0)


def latest_adjudications(con: sqlite3.Connection) -> list[dict]:
    """Latest decision per (task, subject); earlier ones stay in the table as history."""
    rows = con.execute("SELECT * FROM adjudications WHERE id IN (SELECT MAX(id) FROM adjudications "
                       "GROUP BY task_id, subject) ORDER BY task_id, subject").fetchall()
    return [dict(r) for r in rows]
