"""Payload store behaviour: local filesystem, S3 with a fake client, and the runner's failure handling."""
import io
import json
import os
import uuid

import pytest

from src.evalbeacon.core import Case, Output, run_eval
from src.evalbeacon.store import LocalPayloadStore, S3PayloadStore, _safe

KEY = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0"


# ── local ───────────────────────────────────────────────────────────────────
def test_put_get_roundtrip_is_idempotent(tmp_path):
    st = LocalPayloadStore(tmp_path)
    ref = st.put("run1/case1", {"output": "a"})
    assert st.put("run1/case1", {"output": "b"}) == ref  # same key, same ref, overwritten in place
    assert st.get(ref) == {"output": "b"} and len(list(tmp_path.rglob("*.json"))) == 1


@pytest.mark.parametrize("key", ["../../escape", "a/../../b", "/abs/path", "C:\\win\\path", "..", "a//b", "\x00evil", "run/..\\x"])
def test_hostile_keys_stay_inside_root(tmp_path, key):
    st = LocalPayloadStore(tmp_path / "store")
    st.put(key, {"x": 1})
    files = [p for p in (tmp_path).rglob("*") if p.is_file()]
    assert files and all((tmp_path / "store").resolve() in p.resolve().parents for p in files)


def test_get_rejects_refs_outside_the_store(tmp_path):
    (tmp_path / "secret.json").write_text('{"s": 1}')
    st = LocalPayloadStore(tmp_path / "store")
    st.put("a", {"x": 1})
    for ref in ("file:../secret.json", "file:../../secret.json", "s3://b/k", str(tmp_path / "secret.json")):
        with pytest.raises(ValueError):
            st.get(ref)


def test_non_json_values_are_stringified_and_redacted_and_leave_no_temp_files(tmp_path):
    """Regression: a non-serialisable payload used to raise TypeError and leak a .tmp file."""
    st = LocalPayloadStore(tmp_path)
    ref = st.put("r/c", {"blob": f"token {KEY}".encode(), "obj": object(), "nested": {"k": [f"mail a@b.io {KEY}"]}})
    text = (tmp_path / ref.removeprefix("file:")).read_text()
    assert KEY not in text and "a@b.io" not in text
    assert not list(tmp_path.rglob("*.tmp"))


def test_overlong_keys_are_shortened_but_stay_distinct(tmp_path):
    st = LocalPayloadStore(tmp_path)
    r1, r2 = st.put("x" * 400, {"n": 1}), st.put("x" * 399 + "y", {"n": 2})
    assert r1 != r2 and st.get(r1) == {"n": 1} and st.get(r2) == {"n": 2}
    assert all(len(seg) <= 100 for seg in _safe("x" * 400).split("/"))


def test_unavailable_storage_raises_and_runner_records_it_without_leaking(tmp_path):
    blocked = tmp_path / "root_is_a_file"
    blocked.write_text("not a directory")
    cases = [Case("c1", {}, {"must_contain": ["ok"]})]
    run = run_eval("r", lambda _: Output(f"ok {KEY}"), cases, "v", store=LocalPayloadStore(blocked / "sub"))
    r = run["results"][0]
    assert r["status"] == "pass" and r["payload_ref"] is None and r["store_error"]
    assert KEY not in json.dumps(run)  # the evaluation result is intact and clean


def test_store_error_messages_are_redacted():
    class Leaky:
        def put(self, key, payload):
            raise RuntimeError(f"upload failed for credentials {KEY}")

    run = run_eval("r", lambda _: Output("ok"), [Case("c", {}, {})], "v", store=Leaky())
    assert KEY not in json.dumps(run) and run["results"][0]["status"] == "pass"


def test_malformed_payload_inputs_do_not_corrupt_the_case():
    class Circular(dict):
        pass

    c = Circular()
    c["self"] = c  # a circular structure cannot be serialised
    run = run_eval("r", lambda _: Output("ok"), [Case("c", {"q": c}, {})], "v", store=LocalPayloadStore(os.devnull + "/x"))
    assert run["results"][0]["status"] == "pass"  # the run still completes; the failure is a store_error


# ── S3 with a fake client ───────────────────────────────────────────────────
class FakeS3:
    def __init__(self):
        self.objects = {}
        self.fail = None

    def put_object(self, Bucket, Key, Body, ContentType):
        if self.fail:
            raise self.fail
        self.objects[(Bucket, Key)] = Body

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise KeyError("NoSuchKey")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}


def test_s3_put_get_redacts_and_is_idempotent():
    fake = FakeS3()
    st = S3PayloadStore("b", client=fake)
    ref = st.put("run/c1", {"output": KEY})
    assert ref == "s3://b/evalbeacon/run/c1.json" and st.put("run/c1", {"output": KEY}) == ref and len(fake.objects) == 1
    assert KEY not in fake.objects[("b", "evalbeacon/run/c1.json")].decode()
    assert st.get(ref)["output"] != KEY


def test_s3_failures_propagate_and_runner_records_them():
    fake = FakeS3()
    fake.fail = PermissionError("AccessDenied: not authorised to s3:PutObject")
    st = S3PayloadStore("b", client=fake)
    with pytest.raises(PermissionError):
        st.put("r/c", {"a": 1})
    run = run_eval("r", lambda _: Output("ok"), [Case("c", {}, {})], "v", store=st)
    assert "AccessDenied" in run["results"][0]["store_error"] and run["results"][0]["status"] == "pass"
    fake.fail = None
    with pytest.raises(KeyError):
        st.get("s3://b/evalbeacon/missing.json")


def test_s3_get_rejects_refs_outside_bucket_prefix():
    st = S3PayloadStore("b", client=FakeS3())
    for ref in ("s3://other/evalbeacon/x.json", "s3://b/not-our-prefix/x.json", "s3://b/evalbeacon/../x.json", "file:x"):
        with pytest.raises(ValueError):
            st.get(ref)


def test_s3_hostile_keys_stay_under_prefix():
    fake = FakeS3()
    S3PayloadStore("b", prefix="evalbeacon/", client=fake).put("../../other/obj", {"x": 1})
    assert all(k.startswith("evalbeacon/") and ".." not in k for _, k in fake.objects)


# ── real S3: opt-in, disposable bucket only ─────────────────────────────────
BUCKET = os.getenv("EVALBEACON_S3_TEST_BUCKET")


@pytest.mark.skipif(not BUCKET, reason="set EVALBEACON_S3_TEST_BUCKET to a disposable test bucket to run")
def test_real_s3_roundtrip_in_a_disposable_bucket():
    """Writes and deletes only keys under a unique prefix it creates itself. Refuses production-looking buckets."""
    boto3 = pytest.importorskip("boto3")
    assert "prod" not in BUCKET.lower(), "refusing to run integration tests against a production-looking bucket"
    prefix = f"evalbeacon-integration-test/{uuid.uuid4().hex}/"
    client = boto3.client("s3")
    st = S3PayloadStore(BUCKET, prefix=prefix, client=client)
    refs = []
    try:
        refs.append(st.put("run1/case1", {"output": f"leak? {KEY}", "n": 1}))
        assert st.put("run1/case1", {"output": "again"}) == refs[0]
        got = st.get(refs[0])
        assert got == {"output": "again"}
        refs.append(st.put("run1/case2", {"output": f"leak? {KEY}"}))
        assert KEY not in json.dumps(st.get(refs[1]))
    finally:
        for ref in refs:
            client.delete_object(Bucket=BUCKET, Key=ref.removeprefix(f"s3://{BUCKET}/"))
