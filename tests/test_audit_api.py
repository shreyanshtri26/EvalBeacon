"""API behaviour: deployment modes, review-write gating, blinding, and input validation. Offline."""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.audit import api as audit_api
from src.audit import db, tasks

TID = "demo__proj-1"


@pytest.fixture
def env(tmp_path, monkeypatch):
    data = tmp_path / "swebench"
    data.mkdir()
    row = {k: f"x-{k}" for k in tasks.PUBLIC_FIELDS} | {"instance_id": TID, "repo": "demo/proj"}
    (data / "public.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setenv("AUDIT_DATA", str(data))
    monkeypatch.setenv("AUDIT_DB", str(tmp_path / "a.db"))
    monkeypatch.delenv("AUDIT_WRITE", raising=False)
    monkeypatch.setattr(db, "AUDIT_DIR", tmp_path / "audit")
    app = FastAPI()
    app.include_router(audit_api.router)
    app.include_router(audit_api.pages)
    return TestClient(app), db.connect(tmp_path / "a.db")


def ann(**kw):
    return {"task_id": TID, "annotator": "alice", "label": "defect", "category": "underspecified", **kw}


def test_overview_modes(env, monkeypatch, tmp_path):
    c, _ = env
    o = c.get("/api/v1/audit/overview").json()
    assert o["mode"] == "local" and o["tasks"] == 1 and o["write_enabled"] is False
    monkeypatch.setenv("AUDIT_DATA", str(tmp_path / "missing"))
    assert c.get("/api/v1/audit/overview").json()["mode"] == "reports-only"
    assert c.get("/api/v1/audit/tasks").status_code == 404  # no data: say so, do not fake it


def test_writes_are_disabled_by_default(env):
    c, _ = env
    assert c.post("/api/v1/audit/annotations", json=ann()).status_code == 403


def test_write_validation_when_enabled(env, monkeypatch):
    c, _ = env
    monkeypatch.setenv("AUDIT_WRITE", "1")
    assert c.post("/api/v1/audit/annotations", json=ann()).status_code == 201
    assert c.post("/api/v1/audit/annotations", json=ann(label="defect", category="none")).status_code == 422
    assert c.post("/api/v1/audit/annotations", json=ann(task_id="../etc/passwd")).status_code == 422
    verified = {"task_id": TID, "adjudicator": "bob", "status": "verified", "disposition": "false_accept", "reason": "r"}
    assert c.post("/api/v1/audit/adjudications", json=verified).status_code == 422  # verified needs evidence
    ok = c.post("/api/v1/audit/adjudications", json={**verified, "status": "suspect", "disposition": "undetermined"})
    assert ok.status_code == 201


def test_reviewer_only_sees_own_labels_until_they_have_labelled(env, monkeypatch):
    c, con = env
    db.add_annotation(con, task_id=TID, annotator="bob", label="ok", category="none")
    url = f"/api/v1/audit/tasks/{TID}"
    assert c.get(url, params={"as": "alice"}).json()["annotations"] == []
    assert c.get(url, params={"as": "alice", "reveal": 1}).json()["annotations"] == []  # alice has not labelled yet
    db.add_annotation(con, task_id=TID, annotator="alice", label="defect", category="other")
    mine = c.get(url, params={"as": "alice"}).json()["annotations"]
    assert [a["annotator"] for a in mine] == ["alice"]
    both = c.get(url, params={"as": "alice", "reveal": 1}).json()["annotations"]
    assert {a["annotator"] for a in both} == {"alice", "bob"}
    assert c.get(url).json()["annotations"] == []  # anonymous callers see no labels


def test_submission_names_are_aliased_but_controls_are_not(env):
    c, con = env
    base = dict(run_id="r", task_id=TID, repeat=0, grader="g", resolved=1, patch_applied=1, infra_failure=0,
                f2p_passed=1, f2p_total=1, p2p_passed=1, p2p_total=1, failed_tests=[], reason="")
    db.record_run(con, subject="20260217_mini-v2.0.0_claude-4-5-opus-high", **base)
    db.record_run(con, subject="gold", **base)
    runs = c.get(f"/api/v1/audit/tasks/{TID}").json()["runs"]
    assert "20260217" not in json.dumps(runs)
    assert {r["subject_alias"] for r in runs} >= {"subject-" + __import__("hashlib").sha1(b"20260217_mini-v2.0.0_claude-4-5-opus-high").hexdigest()[:6]}
    assert any(r.get("subject") == "gold" for r in runs)


def test_agreement_endpoint(env):
    c, con = env
    for t, la, lb in (("a__b-1", "defect", "defect"), ("a__b-2", "ok", "defect"), ("a__b-3", "ok", "ok"), ("a__b-4", "defect", "ok")):
        db.add_annotation(con, task_id=t, annotator="alice", label=la, category="other" if la == "defect" else "none")
        db.add_annotation(con, task_id=t, annotator="bob", label=lb, category="other" if lb == "defect" else "none")
    r = c.get("/api/v1/audit/agreement", params={"a": "alice", "b": "bob"}).json()
    assert r["items"] == 4 and r["observed_agreement"] == 0.5 and r["kappa_label"] == 0.0 and len(r["disagreements"]) == 2
    assert c.get("/api/v1/audit/agreement", params={"a": "x", "b": "y"}).json()["kappa_label"] is None


def test_report_and_evidence_inputs_are_validated(env):
    c, _ = env
    assert c.get("/api/v1/audit/reports/..%2Fsecret").status_code in (400, 404)
    assert c.get("/api/v1/audit/reports/Bad Name").status_code == 400
    assert c.get("/api/v1/audit/reports/does-not-exist").status_code == 404
    assert c.get("/api/v1/audit/evidence/not-a-hash").status_code == 400
    assert c.get("/api/v1/audit/evidence/" + "0" * 64).status_code == 404


def test_evidence_roundtrip_and_truncation(env):
    c, _ = env
    sha = db.put_evidence("A" * (audit_api.MAX_LOG + 10))
    r = c.get(f"/api/v1/audit/evidence/{sha}").json()
    assert r["truncated"] is True and "[truncated]" in r["log"]


def test_task_list_and_unknown_task(env):
    c, _ = env
    assert c.get("/api/v1/audit/tasks").json()["total"] == 1
    assert c.get("/api/v1/audit/tasks", params={"repo": "other/repo"}).json()["total"] == 0
    assert c.get("/api/v1/audit/tasks/nope__nope-9").status_code == 404
    assert c.get("/api/v1/audit/tasks/not_a_task_id").status_code == 400


def test_audit_page_is_served(env):
    c, _ = env
    r = c.get("/audit")
    assert r.status_code == 200 and "SWE-bench audit" in r.text
