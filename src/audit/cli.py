"""swebench-audit CLI.

  swebench-audit fetch                          download SWE-bench Verified, split public / restricted, show repo shares
  swebench-audit sample --seed S --per-repo N   write a reproducible stratified sample under data/audit/samples
  swebench-audit pull --sample NAME             docker pull the sample's instance images (large: ~3 GB each)
  swebench-audit controls --sample NAME         gold-patch and empty-patch controls through the sandbox
  swebench-audit flake --sample NAME --n 5      gold patch N times per task: verdict disagreement rate (Q5)
  swebench-audit stronger --entry E --sample S  official vs independent tests on a submission's resolved patches (Q1)
  swebench-audit annot-sample / annot-export    blinded annotation sample and packets (Q3)
  swebench-audit agreement --a X --b Y          inter-annotator kappa
  swebench-audit catalogue                      adjudicated defect catalogue
  swebench-audit rescore                        rank change with verified defects removed vs random removal (Q7)
  swebench-audit probe --provider P             repo-less file localisation from issue text (Q4, needs a key)
  swebench-audit summary                        every saved report in one document, Holm-adjusted
  swebench-audit status                         what has been run and reviewed so far

Exit codes: 0 ok · 1 a control or check failed · 2 setup or usage error.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from src.audit import controls, db, experiments, review, sampling, study, submissions, tasks
from src.audit.sandbox import Limits, image_present

SAMPLES = Path("data/audit/samples")
REPORTS = Path("data/audit/reports")


def _sample_ids(name: str) -> list[str]:
    p = SAMPLES / f"{name}.json"
    if not p.exists():
        raise SystemExit(f"error: sample {name!r} not found at {p}; create it with `swebench-audit sample`")
    ids: list[str] = json.loads(p.read_text(encoding="utf-8"))["ids"]
    return ids


def cmd_fetch(_: argparse.Namespace) -> int:
    raw = tasks.fetch_verified()
    m = tasks.split_raw(raw)
    print(json.dumps(m, indent=2))
    for repo, s in tasks.repo_shares(tasks.load_public()).items():
        print(f"  {repo:<28}{s['tasks']:>4}  {s['share']:.1%}")
    return 0


def cmd_sample(a: argparse.Namespace) -> int:
    pub = tasks.load_public()
    ids = sampling.stratified(pub, a.per_repo, a.seed)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    name = a.name or f"{a.seed}-{a.per_repo}-per-repo"
    (SAMPLES / f"{name}.json").write_text(json.dumps(
        {"name": name, "seed": a.seed, "per_stratum": a.per_repo, "stratum": "repo", "dataset": tasks.DATASET,
         "ids": ids}, indent=2), encoding="utf-8")
    print(f"{name}: {len(ids)} tasks -> {SAMPLES / (name + '.json')}")
    return 0


def cmd_pull(a: argparse.Namespace) -> int:
    pub = tasks.load_public()
    for i in _sample_ids(a.sample):
        img = pub[i]["image"]
        free = shutil.disk_usage(".").free / 2**30
        if free < a.min_free_gb:
            print(f"stop: only {free:.0f} GB free (< {a.min_free_gb})")
            return 1
        if image_present(img):
            print(f"have  {i}")
            continue
        print(f"pull  {i} ({free:.0f} GB free)", flush=True)
        if subprocess.run(["docker", "pull", "-q", img], capture_output=True).returncode != 0:
            print(f"FAILED {i}")
            return 1
    return 0


def cmd_controls(a: argparse.Namespace) -> int:
    pub, res = tasks.load_public(), tasks.load_restricted()
    ids = [i for i in _sample_ids(a.sample) if image_present(pub[i]["image"])]
    missing = len(_sample_ids(a.sample)) - len(ids)
    if not ids:
        print("error: no images present; run `swebench-audit pull` first", file=sys.stderr)
        return 2
    if a.limit:
        ids = ids[: a.limit]
    limits = Limits(timeout_s=a.timeout, network=a.network)
    rep = controls.run_controls(ids, pub, res, db.connect(), limits=limits, workers=a.workers, label=a.sample)
    rep["sample_tasks_without_image"] = missing
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"controls-{a.sample}.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    for subject, c in rep["controls"].items():
        print(f" {subject:<6} {c['as_expected']}/{c['cases']} as expected (expected resolved={c['expected_resolved']})")
        for m in c["mismatches"]:
            print(f"   ✗ {m['id']}: {m['status']} {m['reasons']}")
    print(f" controls: {'OK' if rep['ok'] else 'FAILED'}" + (f" ({missing} sample tasks skipped: no image)" if missing else ""))
    return 0 if rep["ok"] else 1


def _save(name: str, obj: dict) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    p = REPORTS / f"{name}.json"
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return p


def cmd_matrix(_: argparse.Namespace) -> int:
    m = experiments.comparability_matrix()
    p = _save("comparability-verified", m)
    print(f"{m['entries']} Verified entries -> {p}")
    for f, n in m["stated"].items():
        print(f"  {f:<20}{n:>4} state it ({n / m['entries']:.0%})")
    return 0


def _mini_entries(since: str = "", until: str = "99999999") -> list:
    names = [n for n in submissions.list_entries() if since <= n[:8] <= until]
    es = [submissions.load_entry(n) for n in names]
    return [e for e in es if e.meta.get("tags", {}).get("agent") == "mini-SWE-agent" and e.trajs_url and e.verdicts]


def cmd_leakstudy(a: argparse.Namespace) -> int:
    pub, res = tasks.load_public(), tasks.load_restricted()
    ids, _ = sampling.proportional(pub, a.tasks, a.seed)
    entries = _mini_entries()
    print(f"scanning {len(entries)} mini-SWE-agent entries x {len(ids)} tasks (seed {a.seed})", flush=True)
    out = experiments.leak_study(entries, ids, res, workers=a.workers)
    out["seed"], out["task_ids"] = a.seed, ids
    p = _save(f"leakstudy-{a.seed}", out)
    for label, g in out["groups"].items():
        r = g["gold_overlap"]
        print(f" {label:<7} entries={g['entries']:<3} scanned={g['scanned']:<5} missing={g['missing']} unparsed={g['unparsed']}"
              f" gold_overlap={r['events'] if r else '-'}/{r['observations'] if r else '-'}"
              + (f" ({r['rate']:.2%}, 95% CI {r['low']:.2%}-{r['high']:.2%}, task-clustered)" if r else ""))
    print(f"-> {p}")
    return 0


def cmd_regrade(a: argparse.Namespace) -> int:
    pub, res = tasks.load_public(), tasks.load_restricted()
    ids = [i for i in _sample_ids(a.sample) if image_present(pub[i]["image"])]
    entry = submissions.load_entry(a.entry)
    out = experiments.regrade(entry, ids, pub, res, db.connect(), limits=Limits(timeout_s=a.timeout, network=a.network),
                              workers=a.workers)
    p = _save(f"regrade-{a.entry}", out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("limits",)}, indent=2))
    print(f"-> {p}")
    return 0


def cmd_nullrank(a: argparse.Namespace) -> int:
    entries = [e for e in (submissions.load_entry(n) for n in submissions.list_entries() if a.since <= n[:8] <= a.until)
               if e.verdicts and e.meta.get("tags", {}).get("agent") == "mini-SWE-agent"]
    out: dict = {"entries": [e.name for e in entries],
                 "null": [experiments.null_rank_change(experiments.outcome_table(entries), k, seed=1) for k in a.k]}
    p = _save(f"nullrank-{a.since}-{a.until}", out)
    for n in out["null"]:
        if n:
            print(f" remove {n['k']:>3} random of {n['tasks']} tasks, {n['agents']} agents: tau mean {n['tau_mean']:.3f}, "
                  f"5th pct {n['tau_p05']:.3f}, min {n['tau_min']:.3f}")
    print(f"-> {p}")
    return 0


def _present(sample: str, pub: dict) -> list[str]:
    return [i for i in _sample_ids(sample) if image_present(pub[i]["image"])]


def cmd_flake(a: argparse.Namespace) -> int:
    pub, res = tasks.load_public(), tasks.load_restricted()
    ids = _present(a.sample, pub)
    if not ids:
        print("error: no images present; run `swebench-audit pull` first", file=sys.stderr)
        return 2
    out = study.flake(ids, pub, res, db.connect(), n=a.n, limits=Limits(timeout_s=a.timeout, network=a.network),
                      label=f"flake-{a.sample}")
    p = _save(f"flake-{a.sample}", out)
    print(f" {out['usable']}/{out['tasks']} tasks usable, flaky {len(out['flaky'])}: {out['flaky']}  -> {p}")
    return 0


def cmd_stronger(a: argparse.Namespace) -> int:
    pub, res = tasks.load_public(), tasks.load_restricted()
    entry = submissions.load_entry(a.entry)
    patches = {}
    for i in _present(a.sample, pub):
        t = submissions.fetch_traj(entry, i)
        if t is not None and t.patch and i in entry.verdicts:
            patches[i] = (t.patch, bool(entry.verdicts[i]))
    out = study.stronger(patches, pub, res, limits=Limits(timeout_s=a.timeout, network=a.network))
    p = _save(f"stronger-{a.entry}", out)
    print(json.dumps({k: v for k, v in out.items() if k != "detail"}, indent=2) + f"\n-> {p}")
    return 0


def cmd_annot_sample(a: argparse.Namespace) -> int:
    s = review.annotation_sample(tasks.load_public(), a.n, a.pilot, a.seed)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    review.write_json(SAMPLES / f"{a.name}.json", {"name": a.name, "stratum": "repo", **s})
    print(f"{a.name}: {s['n']} tasks, pilot {len(s['pilot'])}")
    return 0


def cmd_annot_export(a: argparse.Namespace) -> int:
    pub = tasks.load_public()
    ids = json.loads((SAMPLES / f"{a.sample}.json").read_text(encoding="utf-8"))["pilot" if a.pilot else "ids"]
    p = REPORTS.parent / "packets" / f"{a.sample}{'-pilot' if a.pilot else ''}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(x) + "\n" for x in review.packets(ids, pub)), encoding="utf-8")
    print(f"{len(ids)} blinded packets -> {p}")
    return 0


def cmd_agreement(a: argparse.Namespace) -> int:
    print(json.dumps(review.agreement(db.connect(), a.a, a.b), indent=2))
    return 0


def cmd_catalogue(_: argparse.Namespace) -> int:
    p = _save("defect-catalogue", {"rows": review.catalogue(db.connect())})
    print(f"-> {p}")
    return 0


def cmd_rescore(a: argparse.Namespace) -> int:
    entries = [e for e in (submissions.load_entry(n) for n in submissions.list_entries() if a.since <= n[:8] <= a.until)
               if e.verdicts and e.meta.get("tags", {}).get("agent") == "mini-SWE-agent"]
    out = study.rescore(experiments.outcome_table(entries), db.connect())
    p = _save("rescore", out)
    print(json.dumps(out, indent=2) + f"\n-> {p}")
    return 0


def cmd_probe(a: argparse.Namespace) -> int:
    pub, res = tasks.load_public(), tasks.load_restricted()
    ids, _ = sampling.proportional(pub, a.tasks, a.seed)
    out = study.probe(ids, pub, res, study.provider_ask(a.provider, a.model))
    out.update(provider=a.provider, model=a.model, seed=a.seed)
    p = _save(f"probe-{a.provider}-{a.seed}", out)
    print(f" hit rate {out['hit_rate']:.1%} (95% CI {out['low']:.1%}-{out['high']:.1%}), exact {out['exact']}/{out['tasks']} -> {p}")
    return 0


def cmd_summary(_: argparse.Namespace) -> int:
    p = _save("summary", study.summary(REPORTS, db.connect()))
    print(f"-> {p}")
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    con = db.connect()
    runs = con.execute("SELECT subject, COUNT(*) n, SUM(resolved) r FROM grader_runs GROUP BY subject").fetchall()
    for r in runs:
        print(f" grader runs · {r['subject']:<10} {r['n']:>4} runs, {r['r'] or 0} resolved")
    for t in ("annotations", "adjudications"):
        print(f" {t}: {con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="swebench-audit", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch").set_defaults(fn=cmd_fetch)
    s = sub.add_parser("sample")
    s.add_argument("--seed", required=True)
    s.add_argument("--per-repo", type=int, default=1)
    s.add_argument("--name")
    s.set_defaults(fn=cmd_sample)
    p = sub.add_parser("pull")
    p.add_argument("--sample", required=True)
    p.add_argument("--min-free-gb", type=float, default=12.0)
    p.set_defaults(fn=cmd_pull)
    c = sub.add_parser("controls")
    c.add_argument("--sample", required=True)
    c.add_argument("--limit", type=int)
    c.add_argument("--workers", type=int, default=1)
    c.add_argument("--timeout", type=float, default=1800.0)
    c.add_argument("--network", default="none", choices=["none", "bridge"])
    c.set_defaults(fn=cmd_controls)
    sub.add_parser("matrix", help="comparability matrix over every public Verified submission").set_defaults(fn=cmd_matrix)
    ls = sub.add_parser("leakstudy", help="history-leak scan of public mini-SWE-agent trajectories, before vs after the image fix")
    ls.add_argument("--tasks", type=int, default=100)
    ls.add_argument("--seed", default="leakstudy-v1")
    ls.add_argument("--workers", type=int, default=8)
    ls.set_defaults(fn=cmd_leakstudy)
    rg = sub.add_parser("regrade", help="re-grade a submission's patches in the sandbox and compare with its reported verdicts")
    rg.add_argument("--entry", required=True)
    rg.add_argument("--sample", required=True)
    rg.add_argument("--workers", type=int, default=1)
    rg.add_argument("--timeout", type=float, default=1800.0)
    rg.add_argument("--network", default="none", choices=["none", "bridge"])
    rg.set_defaults(fn=cmd_regrade)
    nr = sub.add_parser("nullrank", help="rank stability when k RANDOM tasks are removed (the noise floor)")
    nr.add_argument("--since", default="20260217")
    nr.add_argument("--until", default="20260226")
    nr.add_argument("--k", type=int, nargs="+", default=[10, 26, 50, 82])
    nr.set_defaults(fn=cmd_nullrank)
    fk = sub.add_parser("flake", help="gold patch N times per task: verdict disagreement rate")
    fk.add_argument("--sample", required=True)
    fk.add_argument("--n", type=int, default=5)
    fk.add_argument("--timeout", type=float, default=1800.0)
    fk.add_argument("--network", default="none", choices=["none", "bridge"])
    fk.set_defaults(fn=cmd_flake)
    st = sub.add_parser("stronger", help="official vs independent tests on a submission's patches")
    st.add_argument("--entry", required=True)
    st.add_argument("--sample", required=True)
    st.add_argument("--timeout", type=float, default=1800.0)
    st.add_argument("--network", default="none", choices=["none", "bridge"])
    st.set_defaults(fn=cmd_stronger)
    asp = sub.add_parser("annot-sample", help="seeded annotation sample with a pilot subset")
    asp.add_argument("--n", type=int, default=100)
    asp.add_argument("--pilot", type=int, default=20)
    asp.add_argument("--seed", default="annot-v1")
    asp.add_argument("--name", default="annot-v1")
    asp.set_defaults(fn=cmd_annot_sample)
    ae = sub.add_parser("annot-export", help="blinded annotation packets (issue text only)")
    ae.add_argument("--sample", default="annot-v1")
    ae.add_argument("--pilot", action="store_true")
    ae.set_defaults(fn=cmd_annot_export)
    ag = sub.add_parser("agreement", help="Cohen kappa between two annotators")
    ag.add_argument("--a", required=True)
    ag.add_argument("--b", required=True)
    ag.set_defaults(fn=cmd_agreement)
    sub.add_parser("catalogue", help="write the adjudicated defect catalogue").set_defaults(fn=cmd_catalogue)
    rs = sub.add_parser("rescore", help="rank change with verified defects removed vs random removal")
    rs.add_argument("--since", default="20260217")
    rs.add_argument("--until", default="20260226")
    rs.set_defaults(fn=cmd_rescore)
    pr = sub.add_parser("probe", help="repo-less file localisation from the issue text (needs a provider key)")
    pr.add_argument("--provider", required=True)
    pr.add_argument("--model")
    pr.add_argument("--tasks", type=int, default=100)
    pr.add_argument("--seed", default="probe-v1")
    pr.set_defaults(fn=cmd_probe)
    sub.add_parser("summary", help="every saved report in one document").set_defaults(fn=cmd_summary)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    a = ap.parse_args(argv)
    try:
        return int(a.fn(a))
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
