"""Pluggable payload stores. Full prompt/output payloads live here, keyed by run/case, not on spans.

Contract for every implementation:
- put() redacts the payload itself, so no store can persist an unredacted payload; values that are not JSON
  (bytes, objects) are stringified and then redacted.
- put() is idempotent: the same key overwrites the same object.
- Keys are sanitised and refs are validated, so neither writes nor reads can leave the store's root/prefix.
- A failure raises; the runner records it as `store_error` on the case result without failing the case.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Protocol

from src.evalbeacon.redact import redact, redact_obj

MAX_SEGMENT = 100


class PayloadStore(Protocol):
    def put(self, key: str, payload: dict) -> str: ...
    def get(self, ref: str) -> dict: ...


def _segment(seg: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]", "_", seg).lstrip(".") or "_"
    if len(clean) > MAX_SEGMENT:  # keep names valid on every filesystem; the hash keeps long keys distinct
        clean = clean[:MAX_SEGMENT - 11] + "-" + hashlib.sha1(seg.encode()).hexdigest()[:10]
    return clean


def _safe(key: str) -> str:
    """Keep each path segment to a safe charset so a case id can never escape the store root."""
    return "/".join(_segment(seg) for seg in key.split("/"))


def _dumps(payload: dict) -> str:
    """Serialise first (nothing is written if this fails). Non-JSON values are stringified, then redacted."""
    return json.dumps(redact_obj(payload), indent=2, default=lambda o: redact(str(o)))


class LocalPayloadStore:
    def __init__(self, root: str | Path = "results/payloads"):
        self.root = Path(root)

    def put(self, key: str, payload: dict) -> str:
        data = _dumps(payload)
        rel = _safe(key) + ".json"
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
            os.replace(tmp, path)  # atomic: readers never see a half-written payload
        except BaseException:
            Path(tmp).unlink(missing_ok=True)  # never leave partial files behind
            raise
        return f"file:{rel}"

    def get(self, ref: str) -> dict:
        if not ref.startswith("file:"):
            raise ValueError(f"not a local payload ref: {ref!r}")
        path = (self.root / ref.removeprefix("file:")).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError(f"ref escapes the payload store: {ref!r}")
        data: dict = json.loads(path.read_text(encoding="utf-8"))
        return data


class S3PayloadStore:
    """S3 / S3-compatible adapter. boto3 is an optional extra (`pip install evalbeacon[s3]`), imported lazily;
    pass `client` to inject a fake. Grant it only s3:PutObject / s3:GetObject on its prefix."""

    def __init__(self, bucket: str, prefix: str = "evalbeacon/", client=None):
        if client is None:
            import boto3
            client = boto3.client("s3")
        self.bucket, self.prefix, self.client = bucket, prefix, client

    def put(self, key: str, payload: dict) -> str:
        k = self.prefix + _safe(key) + ".json"
        self.client.put_object(Bucket=self.bucket, Key=k, Body=_dumps(payload).encode(), ContentType="application/json")
        return f"s3://{self.bucket}/{k}"

    def get(self, ref: str) -> dict:
        head = f"s3://{self.bucket}/{self.prefix}"
        if not ref.startswith(head) or ".." in ref:
            raise ValueError(f"ref is outside this store's bucket/prefix: {ref!r}")
        body = self.client.get_object(Bucket=self.bucket, Key=ref.removeprefix(f"s3://{self.bucket}/"))["Body"].read()
        data: dict = json.loads(body)
        return data
