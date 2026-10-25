"""SWE-bench Verified ingestion. Public task metadata and restricted grading artifacts live in separate files."""
from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path

DATA_DIR = Path("data/swebench")
DATASET = "SWE-bench/SWE-bench_Verified"
_ROWS_URL = ("https://datasets-server.huggingface.co/rows?dataset={ds}&config=default&split=test"
             "&offset={off}&length=100")

# What an agent under test may see. Everything else is answer-bearing or grader-only and must never reach it.
PUBLIC_FIELDS = ("instance_id", "repo", "base_commit", "created_at", "difficulty", "version",
                 "environment_setup_commit", "image", "problem_statement")
RESTRICTED_FIELDS = ("instance_id", "patch", "test_patch", "eval_script", "FAIL_TO_PASS", "PASS_TO_PASS",
                     "hints_text", "log_parser", "eval_type")


def fetch_verified(dest: Path = DATA_DIR, retries: int = 4) -> Path:
    """Download all 500 Verified rows via the Hugging Face datasets-server (stdlib only). Returns the raw file."""
    dest.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for off in range(0, 500, 100):
        for attempt in range(retries):
            try:
                url = _ROWS_URL.format(ds=DATASET.replace("/", "%2F"), off=off)
                with urllib.request.urlopen(url, timeout=60) as r:
                    rows += [x["row"] for x in json.load(r)["rows"]]
                break
            except OSError:
                if attempt == retries - 1:
                    raise
                time.sleep(3)
    if len(rows) != 500:
        raise ValueError(f"expected 500 Verified rows, got {len(rows)}")
    raw = dest / "verified_raw.jsonl"
    raw.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return raw


def split_raw(raw: Path = DATA_DIR / "verified_raw.jsonl", out: Path = DATA_DIR) -> dict:
    """Split the raw dump into public.jsonl and restricted.jsonl; return a manifest with content hashes."""
    rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [r["instance_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate instance_id in raw dump")
    out.mkdir(parents=True, exist_ok=True)
    pub = "".join(json.dumps({k: r[k] for k in PUBLIC_FIELDS}, ensure_ascii=False) + "\n" for r in rows)
    res = "".join(json.dumps({k: r[k] for k in RESTRICTED_FIELDS}, ensure_ascii=False) + "\n" for r in rows)
    (out / "public.jsonl").write_text(pub, encoding="utf-8")
    (out / "restricted.jsonl").write_text(res, encoding="utf-8")
    manifest = {"dataset": DATASET, "tasks": len(rows),
                "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                "public_sha256": hashlib.sha256(pub.encode()).hexdigest(),
                "restricted_sha256": hashlib.sha256(res.encode()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _load(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `swebench-audit fetch` first")
    return {d["instance_id"]: d for d in (json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x)}


def load_public(data: Path = DATA_DIR) -> dict[str, dict]:
    return _load(data / "public.jsonl")


def load_restricted(data: Path = DATA_DIR) -> dict[str, dict]:
    """Grader-only data. Callers outside the grader and audit code must not import this."""
    return _load(data / "restricted.jsonl")


def repo_shares(public: dict[str, dict]) -> dict[str, dict]:
    """Measured, not assumed: task count and share per repository."""
    n = len(public)
    counts: dict[str, int] = {}
    for t in public.values():
        counts[t["repo"]] = counts.get(t["repo"], 0) + 1
    return {r: {"tasks": c, "share": round(c / n, 4)} for r, c in sorted(counts.items(), key=lambda kv: -kv[1])}
