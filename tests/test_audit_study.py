"""Offline tests for flake, stronger-test, rescore, probe and review helpers. No Docker, no network, no model."""
import json

import pytest

from src.audit import db, review, study

PUB = {f"r__a-{i}": {"instance_id": f"r__a-{i}", "repo": "r/a" if i < 3 else "r/b", "base_commit": "c",
                     "problem_statement": f"issue {i}"} for i in range(6)}


@pytest.fixture
def con(tmp_path):
    return db.connect(tmp_path / "a.db")


def test_flake_summary_counts_only_disagreeing_tasks():
    s = study.flake_summary({"a": [True] * 5, "b": [True, False, True, True, True], "c": [False] * 5, "d": [True, None, None]})
    assert s["flaky"] == ["b"] and s["usable"] == 3 and s["always_resolved"] == ["a"] and s["never_resolved"] == ["c"]
    assert s["low"] <= s["rate"] <= s["high"]


def test_extra_script_keeps_official_output_first_and_parses_exit():
    script = study.extra_eval_script("echo official", {"patch": "diff --git a/t.py b/t.py\n", "command": "pytest t.py"})
    assert script.startswith("echo official") and "pytest t.py" in script and "AUDIT_EOF" in script
    assert study.parse_extra("x\nAUDIT_EXTRA_EXIT=0\n") is True
    assert study.parse_extra("AUDIT_EXTRA_EXIT=1") is False
    assert study.parse_extra("AUDIT_EXTRA_EXIT=apply") is None and study.parse_extra("no marker") is None


def test_false_accept_ignores_invalid_extra_tests():
    rows = [{"id": "a", "official": True, "extra": False, "gold_extra": True},    # real false accept
            {"id": "b", "official": True, "extra": True, "gold_extra": True},
            {"id": "c", "official": True, "extra": False, "gold_extra": False},  # extra test rejects gold: invalid
            {"id": "d", "official": False, "extra": False, "gold_extra": True}]
    s = study.false_accept_summary(rows)
    assert s["false_accepts"] == ["a"] and s["official_accepted"] == 2 and s["valid_extra_tests"] == 3


def test_load_extra_requires_patch_and_command(tmp_path):
    (tmp_path / "t1.json").write_text(json.dumps({"patch": "p"}))
    with pytest.raises(ValueError):
        study.load_extra("t1", tmp_path)
    assert study.load_extra("missing", tmp_path) is None


def test_rescore_uses_only_verified_task_defects(con):
    out = {"a1": {f"t{i}": int(i % 2 == 0) for i in range(20)}, "a2": {f"t{i}": int(i % 3 == 0) for i in range(20)},
           "a3": {f"t{i}": int(i < 5) for i in range(20)}}
    db.add_adjudication(con, task_id="t1", adjudicator="r", status="suspect", disposition="task_defect", reason="maybe")
    db.add_adjudication(con, task_id="t2", adjudicator="r", status="verified", disposition="task_defect", reason="x",
                        evidence_ref="sha")
    r = study.rescore(out, con, b=50)
    assert r["excluded"] == ["t2"] and r["ranking"]["excluded_in_common"] == 1 and r["random_removal"]["k"] == 1


def test_probe_scores_file_hits_without_a_model():
    res = {"r__a-0": {"patch": "diff --git a/pkg/x.py b/pkg/x.py\n"}, "r__a-1": {"patch": "diff --git a/pkg/y.py b/pkg/y.py\n"}}
    answers = {"issue 0": "`pkg/x.py`", "issue 1": "pkg/other.py"}
    r = study.probe(["r__a-0", "r__a-1"], PUB, res, lambda p: next(v for k, v in answers.items() if k in p))
    assert r["tasks"] == 2 and [x["hit"] for x in r["rows"]] == [True, False] and r["exact"] == 1


def test_annotation_sample_is_deterministic_and_packets_are_blinded():
    a, b = review.annotation_sample(PUB, 4, 2, "s"), review.annotation_sample(PUB, 4, 2, "s")
    assert a == b and set(a["pilot"]) <= set(a["ids"]) and len(a["pilot"]) == 2
    for p in review.packets(a["ids"], PUB):
        assert set(p) == set(review.PACKET_FIELDS)


def test_agreement_defect_share_and_catalogue(con):
    for t, x, y in (("t1", "defect", "defect"), ("t2", "ok", "defect"), ("t3", "ok", "ok")):
        cat = lambda lab: "underspecified" if lab == "defect" else "none"  # noqa: E731
        db.add_annotation(con, task_id=t, annotator="a", label=x, category=cat(x))
        db.add_annotation(con, task_id=t, annotator="b", label=y, category=cat(y))
    ag = review.agreement(con, "a", "b")
    assert ag["items"] == 3 and len(ag["disagreements"]) == 1 and ag["observed_agreement"] == pytest.approx(2 / 3)
    db.add_adjudication(con, task_id="t1", adjudicator="r", status="verified", disposition="task_defect", reason="x",
                        evidence_ref="e")
    db.add_adjudication(con, task_id="t3", adjudicator="r", status="verified", disposition="correct_accept", reason="y",
                        evidence_ref="e")
    s = review.defect_share(con, ["t1", "t2", "t3"])
    assert s["adjudicated"] == 2 and s["defects"] == 1 and len(review.catalogue(con)) == 2


def test_summary_applies_holm_to_stored_pvalues(tmp_path, con):
    (tmp_path / "x.json").write_text(json.dumps({"mcnemar_p": 0.01, "rows": [1]}))
    (tmp_path / "y.json").write_text(json.dumps({"mcnemar_p": 0.04}))
    s = study.summary(tmp_path, con)
    assert s["holm_adjusted"] == {"x": 0.02, "y": 0.04} and "rows" not in s["reports"]["x"]
