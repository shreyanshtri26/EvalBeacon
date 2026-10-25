"""Statistics for the audit. Stdlib only; all resampling uses an explicit seed. The task is the unit of resampling."""
from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float, float] | None:
    """Wilson score interval for a proportion: (estimate, low, high). None when n == 0 (no denominator)."""
    if n <= 0:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def stratified_proportion(strata: dict[str, tuple[int, int, int]], z: float = Z95) -> dict | None:
    """Population-weighted proportion from a stratified sample.

    strata[name] = (N_h population size, n_h sampled, k_h positives). Uses the standard stratified estimator with a
    finite-population correction; the per-stratum variance uses the add-half-success adjusted proportion so a stratum
    with 0 or n_h positives does not report zero uncertainty. Strata that were not sampled (n_h == 0) are excluded and
    reported in `unsampled_population`, never silently dropped."""
    sampled = {s: v for s, v in strata.items() if v[1] > 0}
    if not sampled:
        return None
    big_n = sum(v[0] for v in sampled.values())
    est, var = 0.0, 0.0
    for pop, n, k in sampled.values():
        w = pop / big_n
        est += w * k / n
        pa = (k + 0.5) / (n + 1)
        fpc = 1 - n / pop if pop > 0 else 0.0
        var += w * w * fpc * pa * (1 - pa) / n
    se = math.sqrt(var)
    return {"estimate": est, "low": max(0.0, est - z * se), "high": min(1.0, est + z * se), "se": se,
            "n": sum(v[1] for v in sampled.values()), "population": big_n,
            "unsampled_population": sum(v[0] for v in strata.values() if v[1] == 0)}


def bootstrap_ci(values: Sequence[float], stat: Callable[[Sequence[float]], float] | None = None, *,
                 b: int = 5000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float] | None:
    """Percentile bootstrap over independent units: (stat, low, high). Default stat is the mean."""
    if not values:
        return None
    f = stat or (lambda xs: sum(xs) / len(xs))
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(f([values[rng.randrange(n)] for _ in range(n)]) for _ in range(b))
    return f(values), draws[int(alpha / 2 * b)], draws[min(b - 1, int((1 - alpha / 2) * b))]


def cluster_ratio_ci(pairs: Sequence[tuple[int, int]], *, b: int = 5000, seed: int = 0) -> dict | None:
    """Rate = sum(k)/sum(n) with a bootstrap that resamples whole clusters (tasks), not individual observations.

    pairs[t] = (events, observations) for task t. Use this when one task contributes many correlated rows
    (for example the same task evaluated under many agents), where a Wilson interval would be too narrow."""
    pairs = [p for p in pairs if p[1] > 0]
    if not pairs:
        return None
    rng = random.Random(seed)
    m = len(pairs)

    def rate(ps: Sequence[tuple[int, int]]) -> float:
        return sum(k for k, _ in ps) / sum(n for _, n in ps)

    draws = sorted(rate([pairs[rng.randrange(m)] for _ in range(m)]) for _ in range(b))
    return {"rate": rate(pairs), "low": draws[int(0.025 * b)], "high": draws[min(b - 1, int(0.975 * b))],
            "clusters": m, "observations": sum(n for _, n in pairs), "events": sum(k for k, _ in pairs)}


def paired_bootstrap_diff(a: dict[str, int], c: dict[str, int], *, b: int = 5000, seed: int = 0) -> dict | None:
    """Difference in resolve rate (c - a) over the tasks both scored, resampling tasks. Missing tasks are reported."""
    ids = sorted(set(a) & set(c))
    if not ids:
        return None
    diffs = [c[i] - a[i] for i in ids]
    r = bootstrap_ci(diffs, b=b, seed=seed)
    assert r is not None
    return {"diff": r[0], "low": r[1], "high": r[2], "n": len(ids),
            "only_in_a": sorted(set(a) - set(c)), "only_in_c": sorted(set(c) - set(a))}


def mcnemar_exact(b01: int, b10: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts (a-fail/c-pass = b01, a-pass/c-fail = b10)."""
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    p: float = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone)."""
    order = sorted(pvals, key=lambda k: pvals[k])
    m, out, running = len(order), {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def cohen_kappa(a: Sequence[str], c: Sequence[str]) -> float | None:
    """Cohen's kappa for two raters over the same items. None when agreement by chance is already 1."""
    if len(a) != len(c) or not a:
        raise ValueError("raters must label the same non-empty item list")
    n = len(a)
    po = sum(x == y for x, y in zip(a, c, strict=True)) / n
    pe = sum((a.count(lbl) / n) * (c.count(lbl) / n) for lbl in set(a) | set(c))
    return None if pe == 1 else (po - pe) / (1 - pe)


def kendall_tau(rank_a: dict[str, float], rank_b: dict[str, float]) -> float | None:
    """Kendall tau-a over items present in both scorings (higher score = better). None below two items."""
    items = sorted(set(rank_a) & set(rank_b))
    if len(items) < 2:
        return None
    conc = disc = 0
    for i, x in enumerate(items):
        for y in items[i + 1:]:
            s = (rank_a[x] - rank_a[y]) * (rank_b[x] - rank_b[y])
            conc += s > 0
            disc += s < 0
    pairs = len(items) * (len(items) - 1) / 2
    return (conc - disc) / pairs


def rank_change_bootstrap(outcomes: dict[str, dict[str, int]], excluded: set[str], *, b: int = 2000,
                          seed: int = 0) -> dict | None:
    """How stable is the agent ranking when `excluded` tasks are removed, resampling tasks?

    outcomes[agent][task] = 1 if resolved. Returns tau between the original and the re-scored ranking plus its
    bootstrap interval. Only tasks every agent has an outcome for are used, and the count is reported."""
    agents = sorted(outcomes)
    common = sorted(set.intersection(*(set(o) for o in outcomes.values()))) if agents else []
    if len(agents) < 2 or not common:
        return None

    def taus(tasks: list[str]) -> float | None:
        full = {a: sum(outcomes[a][t] for t in tasks) / len(tasks) for a in agents}
        kept = [t for t in tasks if t not in excluded]
        if not kept:
            return None
        adj = {a: sum(outcomes[a][t] for t in kept) / len(kept) for a in agents}
        return kendall_tau(full, adj)

    point = taus(common)
    rng = random.Random(seed)
    draws = sorted(t for t in (taus([common[rng.randrange(len(common))] for _ in common]) for _ in range(b))
                   if t is not None)
    return {"tau": point, "tasks_used": len(common), "excluded_in_common": len(excluded & set(common)),
            "low": draws[int(0.025 * len(draws))] if draws else None,
            "high": draws[min(len(draws) - 1, int(0.975 * len(draws)))] if draws else None}
