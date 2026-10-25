"""Known-answer tests for the audit statistics."""
import math

import pytest

from src.audit import sampling, stats


def test_wilson_known_values():
    p, lo, hi = stats.wilson(5, 10)
    assert p == 0.5 and lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    p, lo, hi = stats.wilson(0, 10)
    assert p == 0 and lo == 0 and hi == pytest.approx(0.2775, abs=1e-3)  # zero events still carry uncertainty
    assert stats.wilson(0, 0) is None  # no denominator is not 0%


def test_stratified_proportion_weights_by_population():
    # stratum A: 90 tasks, 1/10 positive; stratum B: 10 tasks, 5/10 positive -> 0.9*0.1 + 0.1*0.5 = 0.14
    r = stats.stratified_proportion({"A": (90, 10, 1), "B": (10, 10, 5)})
    assert r["estimate"] == pytest.approx(0.14) and r["low"] < 0.14 < r["high"]
    # an unsampled stratum is reported, not silently dropped
    r = stats.stratified_proportion({"A": (90, 10, 1), "B": (10, 0, 0)})
    assert r["unsampled_population"] == 10 and r["population"] == 90
    assert stats.stratified_proportion({"A": (5, 0, 0)}) is None


def test_bootstrap_is_deterministic_and_brackets_the_mean():
    xs = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    a, b = stats.bootstrap_ci(xs, seed=7), stats.bootstrap_ci(xs, seed=7)
    assert a == b and a[1] <= a[0] <= a[2] and a[0] == 0.7
    assert stats.bootstrap_ci([]) is None


def test_cluster_ratio_is_wider_than_independent_when_clusters_are_correlated():
    # 10 tasks, each seen by 10 agents; a task is either flagged by all agents or by none (perfect correlation)
    pairs = [(10, 10)] * 3 + [(0, 10)] * 7
    c = stats.cluster_ratio_ci(pairs, seed=2)
    w = stats.wilson(30, 100)
    assert c["rate"] == 0.3 and c["clusters"] == 10 and c["observations"] == 100
    assert (c["high"] - c["low"]) > (w[2] - w[1])  # naive Wilson would overstate precision
    assert stats.cluster_ratio_ci([(0, 0)]) is None


def test_paired_diff_reports_missing_tasks():
    r = stats.paired_bootstrap_diff({"t1": 1, "t2": 0, "t3": 1}, {"t1": 1, "t2": 1, "t4": 0}, seed=1)
    assert r["n"] == 2 and r["diff"] == 0.5 and r["only_in_a"] == ["t3"] and r["only_in_c"] == ["t4"]


def test_mcnemar_exact_and_holm():
    assert stats.mcnemar_exact(0, 0) == 1.0
    assert stats.mcnemar_exact(0, 6) == pytest.approx(2 / 64)  # 2 * P(X<=0 | n=6)
    assert stats.mcnemar_exact(5, 5) == 1.0
    adj = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)


def test_kappa():
    assert stats.cohen_kappa(list("aabb"), list("aabb")) == 1.0
    assert stats.cohen_kappa(list("aabb"), list("abab")) == pytest.approx(0.0)
    assert stats.cohen_kappa(list("aaaa"), list("aaaa")) is None  # chance agreement is total: undefined
    with pytest.raises(ValueError):
        stats.cohen_kappa(["a"], ["a", "b"])


def test_kendall_and_rank_change():
    assert stats.kendall_tau({"x": 3, "y": 2, "z": 1}, {"x": 9, "y": 5, "z": 0}) == 1.0
    assert stats.kendall_tau({"x": 3, "y": 2, "z": 1}, {"x": 0, "y": 5, "z": 9}) == -1.0
    assert stats.kendall_tau({"x": 1}, {"x": 1}) is None
    out = {"A": {"t1": 1, "t2": 1, "t3": 0, "t4": 0}, "B": {"t1": 1, "t2": 0, "t3": 1, "t4": 1}}
    # removing t3,t4 flips the order of A and B: B led 3/4 vs 2/4 before, trails 1/2 vs 2/2... A=1.0,B=0.5
    r = stats.rank_change_bootstrap(out, {"t3", "t4"}, b=200, seed=3)
    assert r["tau"] == -1.0 and r["tasks_used"] == 4 and r["excluded_in_common"] == 2
    assert stats.rank_change_bootstrap({"A": {"t": 1}}, set()) is None


def test_sampling_is_deterministic_and_stratified():
    pub = {f"r{i % 3}__t{i}": {"repo": f"r{i % 3}"} for i in range(30)}
    a, b = sampling.stratified(pub, 2, "s"), sampling.stratified(pub, 2, "s")
    assert a == b and len(a) == 6 and len({pub[i]["repo"] for i in a}) == 3
    assert a != sampling.stratified(pub, 2, "other-seed")
    ids, w = sampling.proportional(pub, 9, "s")
    assert len(ids) == 9 and all(w[i] == pytest.approx(10 / 3) for i in ids)
    assert math.isclose(sum(w.values()), 30)  # weights reconstruct the population
