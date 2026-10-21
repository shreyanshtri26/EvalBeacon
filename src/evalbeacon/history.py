"""Run history with idempotent saves and count-based retention."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

ROOT = Path("results/history")


def save_run(run: dict, root: Path = ROOT, keep: int = 50) -> Path:
    """Write the run keyed by run_id (re-saving the same run overwrites, never duplicates), then prune."""
    root.mkdir(parents=True, exist_ok=True)
    existing = next(iter(root.glob(f"*_{run['run_id']}.json")), None)
    path = existing or root / f"{int(run['started_at'])}_{run['run_id']}.json"
    fd, tmp = tempfile.mkstemp(dir=root, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(run, f)
    os.replace(tmp, path)
    for old in sorted(root.glob("*.json"))[:-keep] if keep > 0 else []:
        old.unlink()
    return path


def summarize(r: dict) -> dict:
    return {"run_id": r["run_id"], "name": r["name"], "status": r["status"], "started_at": r["started_at"],
            "dataset_version": r["dataset_version"], **{k: r["summary"][k] for k in ("cases", "pass_rate", "error_rate")}}


def list_runs(root: Path = ROOT) -> list[dict]:
    """Newest first; one summary row per saved run."""
    rows = []
    for p in sorted(root.glob("*.json"), reverse=True):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
            rows.append(summarize(r))
        except (OSError, ValueError, KeyError):
            continue  # a corrupt history file must not break listing
    return rows
