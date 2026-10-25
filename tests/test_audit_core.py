"""Offline tests for the audit modules: data split, leakage scan, review workflow rules. No Docker, no network."""
import json

import pytest

from src.audit import db, submissions, tasks

GOLD = "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x\n+    return compute_separable_matrix(left, right)\n"
MSG = "mini-swe-agent-1.1"


def traj(*cmds):
    return submissions.Traj("t", "", None, list(cmds), MSG)


def test_public_view_never_contains_answer_bearing_fields(tmp_path):
    row = {k: f"v-{k}" for k in set(tasks.PUBLIC_FIELDS) | set(tasks.RESTRICTED_FIELDS)}
    row.update(instance_id="a__b-1", FAIL_TO_PASS=["t1"], PASS_TO_PASS=["t2"])
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps(row) + "\n", encoding="utf-8")
    m = tasks.split_raw(raw, tmp_path)
    pub = tasks.load_public(tmp_path)["a__b-1"]
    assert m["tasks"] == 1 and set(pub) == set(tasks.PUBLIC_FIELDS)
    for hidden in ("patch", "test_patch", "eval_script", "FAIL_TO_PASS", "PASS_TO_PASS", "hints_text"):
        assert hidden not in pub


def test_duplicate_ids_are_rejected(tmp_path):
    raw = tmp_path / "raw.jsonl"
    row = {k: "x" for k in set(tasks.PUBLIC_FIELDS) | set(tasks.RESTRICTED_FIELDS)}
    raw.write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        tasks.split_raw(raw, tmp_path)


def test_agents_own_git_diff_is_not_leakage():
    # regression: an agent that fixes the bug the same way as the gold patch sees the gold line in `git diff`
    own = traj(("cd /testbed && git diff f.py", "diff --git a/f.py b/f.py\n+    return compute_separable_matrix(left, right)\n"))
    r = submissions.leak_scan(own, GOLD)
    assert r["level"] == "none" and not r["gold_overlap"]


def test_gold_lines_in_history_output_are_flagged():
    leak = traj(("git log --all -p -3", "commit abc\n+    return compute_separable_matrix(left, right)\n"))
    r = submissions.leak_scan(leak, GOLD)
    assert r["level"] == "gold_overlap" and r["gold_overlap"][0]["matched_lines"] == 1 and r["broad_history"]


def test_history_without_gold_overlap_is_graded_lower():
    assert submissions.leak_scan(traj(("git log --oneline -5", "abc fix")), GOLD)["level"] == "history_commands"
    assert submissions.leak_scan(traj(("git log --all", "abc")), GOLD)["level"] == "broad_history"
    assert submissions.leak_scan(traj(("ls", "")), GOLD)["level"] == "none"


def test_context_lines_do_not_count_as_added_lines():
    unchanged = traj(("git show HEAD", " compute_separable_matrix(left, right)_unchanged_context_line_here\n"))
    assert submissions.leak_scan(unchanged, GOLD)["gold_overlap"] == []


def test_parse_traj_rejects_unknown_format():
    with pytest.raises(ValueError, match="unsupported"):
        submissions.parse_traj({"trajectory_format": "other-agent-3"})


def test_parse_traj_pairs_outputs_with_commands():
    raw = {"trajectory_format": MSG, "instance_id": "t", "info": {"submission": "P", "exit_status": "Submitted"},
           "messages": [{"role": "assistant", "tool_calls": [{"id": "c1", "function": {"arguments": json.dumps({"command": "ls"})}}]},
                        {"role": "tool", "tool_call_id": "c1", "content": "<o>", "extra": {"raw_output": "file.py\n"}}]}
    t = submissions.parse_traj(raw)
    assert t.patch == "P" and t.commands == [("ls", "file.py\n")]


def test_parse_legacy_traj_extracts_bash_blocks_and_next_user_output():
    raw = {"info": {"submission": "P", "exit_status": "Submitted"}, "instance_id": "t", "messages": [
        {"role": "system", "content": "s"},
        {"role": "assistant", "content": "THOUGHT: look\n\n```bash\ngit log --all -p -2\n```"},
        {"role": "user", "content": [{"type": "text", "text": "<output>\n+    return compute_separable_matrix(left, right)\n</output>"}]},
        {"role": "assistant", "content": "no command here"}]}
    t = submissions.parse_traj(raw)
    assert t.fmt == "mini-swe-agent-legacy" and [c for c, _ in t.commands] == ["git log --all -p -2"]
    assert submissions.leak_scan(t, GOLD)["level"] == "gold_overlap"


@pytest.fixture
def con(tmp_path):
    return db.connect(tmp_path / "a.db")


def test_review_rules(con):
    with pytest.raises(ValueError):
        db.add_annotation(con, task_id="t", annotator="a", label="defect", category="none")
    with pytest.raises(ValueError):
        db.add_annotation(con, task_id="t", annotator="a", label="maybe", category="other")
    db.add_annotation(con, task_id="t", annotator="a", label="defect", category="underspecified")
    with pytest.raises(ValueError, match="evidence_ref"):
        db.add_adjudication(con, task_id="t", adjudicator="r", status="verified", disposition="false_accept", reason="x")
    with pytest.raises(ValueError, match="reason"):
        db.add_adjudication(con, task_id="t", adjudicator="r", status="suspect", disposition="undetermined", reason=" ")


def test_latest_adjudication_wins_but_history_is_kept(con):
    db.add_adjudication(con, task_id="t", adjudicator="r", status="suspect", disposition="undetermined", reason="first look")
    db.add_adjudication(con, task_id="t", adjudicator="r", status="verified", disposition="false_accept",
                        reason="stronger test fails", evidence_ref="sha")
    latest = db.latest_adjudications(con)
    assert len(latest) == 1 and latest[0]["status"] == "verified"
    assert con.execute("SELECT COUNT(*) FROM adjudications").fetchone()[0] == 2


def test_evidence_is_content_addressed_and_idempotent(tmp_path):
    a = db.put_evidence("log text", tmp_path)
    assert a == db.put_evidence("log text", tmp_path) and db.get_evidence(a, tmp_path) == "log text"
    assert len(list(tmp_path.glob("*.log.gz"))) == 1


def test_get_retries_transient_errors(monkeypatch):
    import io
    import urllib.error

    from src.audit import submissions

    calls = []

    class R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    def flaky(req, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise urllib.error.URLError("reset")
        return R(b"ok")

    monkeypatch.setattr(submissions.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(submissions.time, "sleep", lambda s: None)
    assert submissions._get("https://example.test/x") == "ok" and len(calls) == 3
