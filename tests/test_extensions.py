"""Optional extensions: payload store, history, judge, similarity, drift, replay, custom provider."""
import json
from pathlib import Path

import pytest

from src.evalbeacon import history
from src.evalbeacon.cli import build_report, main
from src.evalbeacon.core import Case, Output, load_dataset, run_eval
from src.evalbeacon.demo import kb_search, toolbox_baseline
from src.evalbeacon.drift import detect, load_current
from src.evalbeacon.judge import JudgeReply, parse_reply
from src.evalbeacon.replay import (
    Cassette,
    Recorder,
    Replayer,
    ReplayMissError,
    SideEffectBlockedError,
)
from src.evalbeacon.similarity import similarity
from src.evalbeacon.store import LocalPayloadStore, S3PayloadStore

ROOT = Path(__file__).resolve().parents[1]
DATA = str(ROOT / "datasets/support.jsonl")
KEY = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0"


# ── payload store ───────────────────────────────────────────────────────────
def test_store_redacts_before_persisting_and_blocks_path_traversal(tmp_path):
    st = LocalPayloadStore(tmp_path)
    ref = st.put("run/../../evil", {"output": f"key {KEY}", "nested": [{"m": "a@b.io"}]})
    assert KEY not in (tmp_path / ref.removeprefix("file:")).read_text() and "a@b.io" not in json.dumps(st.get(ref))
    assert all(tmp_path in p.parents for p in tmp_path.rglob("*.json"))  # nothing escaped the root


def test_runner_stores_payload_and_survives_store_outage(tmp_path):
    cases = [Case("c1", {"q": f"my key {KEY}"}, {"must_contain": ["ok"]})]
    r = run_eval("r", lambda _: Output(f"ok {KEY}"), cases, "v", store=LocalPayloadStore(tmp_path), run_id="run1")
    res = r["results"][0]
    assert KEY not in json.dumps(LocalPayloadStore(tmp_path).get(res["payload_ref"]))

    class Down:
        def put(self, *_):
            raise OSError("disk full")

    r = run_eval("r", lambda _: Output("ok"), cases, "v", store=Down())
    assert r["results"][0]["status"] == "pass" and "disk full" in r["results"][0]["store_error"]


def test_s3_store_with_fake_client():
    objs = {}

    class Fake:
        def put_object(self, Bucket, Key, Body, ContentType):
            objs[Key] = Body

        def get_object(self, Bucket, Key):
            import io
            return {"Body": io.BytesIO(objs[Key])}

    st = S3PayloadStore("b", client=Fake())
    ref = st.put("run/c1", {"output": KEY})
    assert ref.startswith("s3://b/evalbeacon/") and KEY not in objs[ref.removeprefix("s3://b/")].decode()
    assert st.get(ref)["output"] != KEY


# ── history ─────────────────────────────────────────────────────────────────
def test_history_is_idempotent_and_pruned(tmp_path):
    cases, v = load_dataset(DATA)
    runs = [run_eval(f"r{i}", lambda _: Output("x"), cases[:1], v, run_id=f"id{i}") for i in range(4)]
    for r in runs:
        history.save_run(r, tmp_path, keep=3)
        history.save_run(r, tmp_path, keep=3)  # re-saving the same run must not duplicate
    assert len(list(tmp_path.glob("*.json"))) == 3
    (tmp_path / "0_corrupt.json").write_text("{nope")
    assert len(history.list_runs(tmp_path)) == 3  # corrupt file skipped, not fatal


# ── judge (optional, never the only check) ──────────────────────────────────
def _judge(score):
    return lambda prompt: JudgeReply(f'noise {{"score": {score}, "rationale": "ok"}}', "gpt-4o-mini", 100, 20)


JUDGED = Case("j1", {"question": "q"}, {"must_contain": ["answer"], "judge": {"rubric": "polite", "min_score": 4}})


def test_judge_records_model_cost_uncertainty_and_gates():
    ok = run_eval("r", lambda _: Output("an answer"), [JUDGED], "v", judge=_judge(5))
    j = ok["results"][0]["judge"]
    assert ok["results"][0]["status"] == "pass" and j["judge_model"] == "gpt-4o-mini" and j["cost_usd"] > 0
    assert "subjective" in j["uncertainty"] and ok["summary"]["judge_cost_usd"] > 0
    low = run_eval("r", lambda _: Output("an answer"), [JUDGED], "v", judge=_judge(2))["results"][0]
    assert low["status"] == "fail" and "judge: score 2" in low["reasons"][0]


def test_judge_cannot_rescue_a_deterministic_failure_and_needs_configuration():
    r = run_eval("r", lambda _: Output("nothing relevant"), [JUDGED], "v", judge=_judge(5))["results"][0]
    assert r["status"] == "fail" and any("must_contain" in x for x in r["reasons"])
    assert run_eval("r", lambda _: Output("an answer"), [JUDGED], "v")["results"][0]["status"] == "error"


def test_judge_sees_redacted_output_and_bad_replies_are_errors():
    seen = []
    run_eval("r", lambda _: Output(f"an answer {KEY}"), [JUDGED], "v",
             judge=lambda p: seen.append(p) or JudgeReply('{"score":5}'))
    assert KEY not in seen[0]
    for bad in ("no json here", '{"score": 9}', '{"score": true}'):
        with pytest.raises(ValueError):
            parse_reply(bad)


# ── similarity ──────────────────────────────────────────────────────────────
def test_similarity_lexical_and_embedding():
    assert similarity("return policy 30 days", "30 days return policy") == 1.0
    assert similarity("alpha beta", "gamma delta") == 0.0
    assert similarity("a", "b", embed_fn=lambda t: [[1.0, 0.0], [1.0, 0.0]]) == 1.0
    case = Case("s", {}, {"similar_to": {"text": "return policy thirty days", "min": 0.9}})
    assert run_eval("r", lambda _: Output("totally unrelated"), [case], "v")["results"][0]["status"] == "fail"


# ── drift ───────────────────────────────────────────────────────────────────
def test_drift_stable_vs_drifted_and_edge_cases(tmp_path, monkeypatch):
    golden = build_report(DATA, "candidate-good", None)["baseline"]
    stable = detect(golden, load_current(ROOT / "datasets/prod_sample_stable.jsonl"))
    drifted = detect(golden, load_current(ROOT / "datasets/prod_sample_drifted.jsonl"))
    assert stable["decision"] == "stable" and stable["exit_code"] == 0
    assert drifted["decision"] == "drift" and drifted["exit_code"] == 1 and drifted["drifted"]
    few = detect(golden, {"refund-001": "x"}, min_cases=5)
    assert few["decision"] == "inconclusive" and few["exit_code"] == 3
    bad = tmp_path / "p.jsonl"
    bad.write_text("not json\n")
    with pytest.raises(ValueError, match=":1:"):
        load_current(bad)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results/baseline.json").write_text(json.dumps(golden))
    assert main(["drift", "--current", str(ROOT / "datasets/prod_sample_drifted.jsonl")]) == 1
    assert main(["drift", "--current", "missing.jsonl"]) == 2


# ── replay with mocked tools ────────────────────────────────────────────────
def test_replay_reproduces_without_live_tools_and_blocks_side_effects(tmp_path, monkeypatch):
    cases, v = load_dataset(DATA)
    cassette = Cassette()
    rec = lambda: Recorder({"kb_search": kb_search}, cassette)  # noqa: E731
    first = run_eval("rec", lambda i: toolbox_baseline(i, rec()), cases, v, workers=1)
    cassette.save(tmp_path / "c.json")

    loaded = Cassette.load(tmp_path / "c.json")

    def poisoned(_q):
        raise AssertionError("live tool called during replay")

    monkeypatch.setattr("src.evalbeacon.demo._lookup", poisoned)  # any live lookup now blows up
    second = run_eval("rep", lambda i: toolbox_baseline(i, Replayer(loaded)), cases, v, workers=1)
    assert [r["output"] for r in first["results"]] == [r["output"] for r in second["results"]]
    with pytest.raises(ReplayMissError):
        Replayer(loaded).call("kb_search", question="never recorded")
    with pytest.raises(SideEffectBlockedError):
        Recorder({"issue_refund": lambda **_: 1}, Cassette(), side_effects={"issue_refund"}).call("issue_refund", id=1)
    (tmp_path / "bad.json").write_text('{"version": 9}')
    with pytest.raises(ValueError, match="unsupported"):
        Cassette.load(tmp_path / "bad.json")


def test_cli_history_replay_and_store(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    pol = str(ROOT / "configs/gate.json")
    assert main(["demo", "--dataset", DATA, "--policy", pol, "--store", "--keep", "2"]) == 1
    assert len(list((tmp_path / "results/history").glob("*.json"))) == 2
    assert any((tmp_path / "results/payloads").rglob("*.json"))
    assert main(["history"]) == 0 and "2 run(s)" in capsys.readouterr().out
    assert main(["replay"]) == 2  # no cassette recorded yet: clean setup error
    assert main(["replay", "--record", "--dataset", DATA]) == 0
    assert main(["replay", "--dataset", DATA]) == 0


def test_custom_provider_is_available_via_env(monkeypatch):
    import importlib

    from src.collectors import providers
    monkeypatch.setenv("EVALBEACON_CUSTOM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("EVALBEACON_CUSTOM_MODEL", "llama3")
    importlib.reload(providers)
    try:
        assert providers.PROVIDERS["custom"][0] == "http://localhost:11434/v1"
        assert providers.default_model("custom") == "llama3"
    finally:
        monkeypatch.delenv("EVALBEACON_CUSTOM_BASE_URL")
        importlib.reload(providers)
