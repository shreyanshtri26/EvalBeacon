"""Studies that need the sandbox or a model: flake reruns (Q5), stronger tests (Q1), rescore (Q7), localisation probe (Q4)."""
from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path

from src.audit import db, grader, review, sandbox, stats
from src.audit.experiments import null_rank_change

EXTRA_DIR = Path("data/audit/extra_tests")
MARK = "AUDIT_EXTRA_EXIT="


# ---- Q5: rerun the same patch N times and count tasks whose verdict changes -------------------------------------
def flake_summary(verdicts: dict[str, list[bool | None]]) -> dict:
    """verdicts[task] = one entry per rerun (None = infrastructure error, ignored). Disagreement = both outcomes seen."""
    usable = {t: [v for v in vs if v is not None] for t, vs in verdicts.items()}
    usable = {t: vs for t, vs in usable.items() if len(vs) >= 2}
    flaky = sorted(t for t, vs in usable.items() if len(set(vs)) > 1)
    ci = stats.wilson(len(flaky), len(usable)) if usable else None
    return {"tasks": len(verdicts), "usable": len(usable), "flaky": flaky,
            "rate": ci[0] if ci else None, "low": ci[1] if ci else None, "high": ci[2] if ci else None,
            "always_resolved": sorted(t for t, vs in usable.items() if all(vs)),
            "never_resolved": sorted(t for t, vs in usable.items() if not any(vs))}


def flake(ids: list[str], public: dict[str, dict], restricted: dict[str, dict], con: sqlite3.Connection, *,
          n: int = 5, limits: sandbox.Limits = sandbox.Limits(), label: str = "flake") -> dict:
    """Grade the gold patch n times per task in fresh containers; every run is stored with its repeat index."""
    out: dict[str, list[bool | None]] = {}
    for i in ids:
        out[i] = []
        t, r = public[i], restricted[i]
        for rep in range(n):
            sr = sandbox.run_patch(t["image"], r["eval_script"], r["patch"], limits)
            if sr.error:
                out[i].append(None)
                continue
            v = grader.grade(t, r, sr.log, r["patch"])
            db.record_run(con, run_id=label, task_id=i, subject="gold", repeat=rep, grader=v.grader,
                          resolved=int(v.resolved), patch_applied=int(v.patch_applied), infra_failure=int(v.infra_failure),
                          f2p_passed=v.f2p_passed, f2p_total=v.f2p_total, p2p_passed=v.p2p_passed, p2p_total=v.p2p_total,
                          failed_tests=v.failed_tests, reason=v.reason, patch_sha256=sr.patch_sha256,
                          eval_script_sha256=sr.eval_script_sha256, log_sha256=db.put_evidence(sr.log), image=sr.image,
                          image_id=sr.image_id, docker_server=sr.docker_server, network=limits.network,
                          memory=limits.memory, cpus=limits.cpus, timeout_s=limits.timeout_s,
                          duration_s=round(sr.duration_s, 1), timed_out=int(sr.timed_out), error=None)
            out[i].append(v.resolved)
    return {"label": label, "repeats": n, "limits": limits.__dict__, **flake_summary(out)}


# ---- Q1: independent, stronger tests ---------------------------------------------------------------------------
def load_extra(task_id: str, root: Path = EXTRA_DIR) -> dict | None:
    """Stronger tests for one task: {"patch": diff adding NEW test files, "command": shell command; exit 0 = pass}."""
    f = root / f"{task_id}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    if not d.get("patch") or not d.get("command"):
        raise ValueError(f"{f}: needs both 'patch' and 'command'")
    return d  # type: ignore[no-any-return]


def extra_eval_script(eval_script: str, extra: dict) -> str:
    """Official eval script, then the independent tests. The extra output sits after the official markers, so the
    official grader still sees exactly what it normally sees."""
    return (f"{eval_script}\ncd /testbed\ncat > /tmp/extra.diff <<'AUDIT_EOF'\n{extra['patch']}\nAUDIT_EOF\n"
            f"if git apply /tmp/extra.diff; then ( {extra['command']} ); echo {MARK}$?; else echo {MARK}apply; fi\n")


def parse_extra(log: str) -> bool | None:
    """True/False for pass/fail of the independent tests; None when they never ran or the patch did not apply."""
    m = re.findall(rf"{MARK}(\w+)", log)
    return None if not m or not m[-1].isdigit() else m[-1] == "0"


def false_accept_summary(rows: list[dict]) -> dict:
    """rows: {"id", "official": bool, "extra": bool|None, "gold_extra": bool|None}. Only tasks whose independent tests
    pass for the gold patch count (an invalid extra test must not manufacture false accepts)."""
    valid = [r for r in rows if r["gold_extra"] is True and r["extra"] is not None]
    accepted = [r for r in valid if r["official"]]
    fa = [r["id"] for r in accepted if not r["extra"]]
    ci = stats.wilson(len(fa), len(accepted)) if accepted else None
    return {"rows": len(rows), "valid_extra_tests": len(valid), "official_accepted": len(accepted),
            "false_accepts": sorted(fa), "rate": ci[0] if ci else None, "low": ci[1] if ci else None,
            "high": ci[2] if ci else None}


def stronger(entry_patches: dict[str, tuple[str, bool]], public: dict[str, dict], restricted: dict[str, dict], *,
             limits: sandbox.Limits = sandbox.Limits(), root: Path = EXTRA_DIR) -> dict:
    """entry_patches[task] = (patch, official verdict). Runs each against the official and the independent tests,
    plus the gold patch against the independent tests as the validity control."""
    rows, skipped = [], []
    for i, (patch, official) in entry_patches.items():
        extra = load_extra(i, root)
        if extra is None:
            skipped.append(i)
            continue
        t, r = public[i], restricted[i]
        script = extra_eval_script(r["eval_script"], extra)
        cand = sandbox.run_patch(t["image"], script, patch, limits)
        gold = sandbox.run_patch(t["image"], script, r["patch"], limits)
        rows.append({"id": i, "official": official, "extra": None if cand.error else parse_extra(cand.log),
                     "gold_extra": None if gold.error else parse_extra(gold.log),
                     "candidate_log": db.put_evidence(cand.log), "gold_log": db.put_evidence(gold.log)})
    return {"no_extra_tests": skipped, **false_accept_summary(rows), "detail": rows}


# ---- Q7: re-score with adjudicated defects removed, against the random-removal noise floor --------------------------
def rescore(outcomes: dict[str, dict[str, int]], con: sqlite3.Connection, *, b: int = 2000, seed: int = 0) -> dict:
    """Ranking change when verified task defects are removed, next to the change from removing the same number of
    RANDOM tasks. Only a tau clearly below the random-removal 5th percentile is evidence the defects matter."""
    excluded = {r["task_id"] for r in db.latest_adjudications(con)
                if r["status"] == "verified" and r["disposition"] == "task_defect"}
    real = stats.rank_change_bootstrap(outcomes, excluded, b=b, seed=seed)
    k = real["excluded_in_common"] if real else 0
    null = null_rank_change(outcomes, k, seed=seed) if k else None
    beyond = bool(real and null and real["tau"] is not None and real["tau"] < null["tau_p05"])
    return {"excluded": sorted(excluded), "ranking": real, "random_removal": null, "beyond_noise_floor": beyond}


# ---- Q4: repo-less localisation probe (needs a model) --------------------------------------------------------------
def gold_files(patch: str) -> set[str]:
    return set(re.findall(r"^diff --git a/(\S+) b/", patch, flags=re.M))


def parse_files(answer: str) -> set[str]:
    return {m.strip("`'\" ,") for m in re.findall(r"[\w./-]+\.\w+", answer)}


def probe(ids: list[str], public: dict[str, dict], restricted: dict[str, dict], ask: Callable[[str], str]) -> dict:
    """Ask a model to name the files to edit from the issue text ALONE (no repo, no tools). A high hit rate on
    tasks the model could not infer from the text is a memorisation signal. `ask` is injected so tests need no model."""
    rows = []
    for i in ids:
        prompt = (f"Repository: {public[i]['repo']}\n\nIssue:\n{public[i]['problem_statement']}\n\n"
                  "You cannot see the repository. List the file paths you would edit to fix this, one per line, no prose.")
        guess, gold = parse_files(ask(prompt)), gold_files(restricted[i]["patch"])
        rows.append({"id": i, "repo": public[i]["repo"], "hit": bool(guess & gold), "exact": guess == gold,
                     "guess": sorted(guess), "gold": sorted(gold)})
    hits = sum(r["hit"] for r in rows)
    ci = stats.wilson(hits, len(rows)) if rows else None
    return {"tasks": len(rows), "hit_rate": ci[0] if ci else None, "low": ci[1] if ci else None,
            "high": ci[2] if ci else None, "exact": sum(r["exact"] for r in rows), "rows": rows}


def provider_ask(provider: str, model: str | None = None) -> Callable[[str], str]:
    from src.collectors import providers
    client, m = providers.make_client(provider), model or providers.default_model(provider)

    def ask(prompt: str) -> str:
        r = client.chat.completions.create(model=m, messages=[{"role": "user", "content": prompt}],
                                           **providers.chat_kwargs(provider, m, 300))
        return r.choices[0].message.content or ""
    return ask


# ---- summary ---------------------------------------------------------------------------------------------------
def summary(reports: Path, con: sqlite3.Connection) -> dict:
    """Collect every saved report and the review state into one document; Holm over the p-values that exist."""
    out: dict = {"reports": {}, "review": {"adjudicated": len(review.catalogue(con))}}
    pvals: dict[str, float] = {}
    for f in sorted(reports.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        out["reports"][f.stem] = ({k: v for k, v in d.items() if k not in ("rows", "detail", "task_ids")}
                                  if isinstance(d, dict) else d)
        if isinstance(d, dict) and isinstance(d.get("mcnemar_p"), float):
            pvals[f.stem] = d["mcnemar_p"]
    out["holm_adjusted"] = stats.holm(pvals)
    return out
