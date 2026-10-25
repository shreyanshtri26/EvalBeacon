"""Isolated patch evaluation: one throwaway container per run, bounded resources, no host mounts."""
from __future__ import annotations

import hashlib
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from swebench.harness.constants import APPLY_PATCH_FAIL, APPLY_PATCH_PASS


@dataclass(frozen=True)
class Limits:
    memory: str = "3g"
    cpus: str = "4"
    pids: int = 1024
    timeout_s: float = 1800.0
    network: str = "none"  # the upstream harness runs with network; the policy is recorded on every result


@dataclass
class SandboxResult:
    log: str
    timed_out: bool
    duration_s: float
    image: str
    image_id: str | None
    patch_sha256: str
    eval_script_sha256: str
    limits: dict
    docker_server: str | None = None
    error: str | None = None  # infrastructure failure, distinct from a failing test


def _sh(args: list[str], timeout: float | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=timeout, check=check)


def _wrapper(has_patch: bool) -> str:
    """Apply the candidate patch the way the upstream harness does (git apply, then patch --fuzz), then grade."""
    apply = ""
    if has_patch:
        apply = (
            "cd /testbed\n"
            "if git apply --verbose /tmp/patch.diff; then echo '" + APPLY_PATCH_PASS + "'\n"
            "elif patch --batch --fuzz=5 -p1 -i /tmp/patch.diff; then echo '" + APPLY_PATCH_PASS + "'\n"
            "else echo '" + APPLY_PATCH_FAIL + "'; exit 1; fi\n"
        )
    return "#!/bin/bash\n" + apply + "bash /tmp/eval.sh\n"


def image_present(image: str) -> bool:
    return _sh(["docker", "image", "inspect", image], check=False).returncode == 0


def run_patch(image: str, eval_script: str, patch: str, limits: Limits = Limits()) -> SandboxResult:
    """Apply `patch` (may be empty) inside a fresh container of `image` and run the task's eval script."""
    name = "audit-" + uuid.uuid4().hex[:12]
    t0 = time.time()
    meta: dict[str, Any] = dict(image=image, patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),
                eval_script_sha256=hashlib.sha256(eval_script.encode()).hexdigest(), limits=asdict(limits))
    try:
        info = _sh(["docker", "version", "--format", "{{.Server.Version}}"], timeout=30).stdout.strip()
        image_id = _sh(["docker", "image", "inspect", "--format", "{{.Id}}", image], timeout=30).stdout.strip()
    except (subprocess.SubprocessError, OSError) as e:
        return SandboxResult("", False, time.time() - t0, image_id=None, error=f"docker unavailable: {e}", **meta)
    try:
        _sh(["docker", "create", "--name", name, "--network", limits.network, "--memory", limits.memory,
             "--memory-swap", limits.memory, "--cpus", limits.cpus, "--pids-limit", str(limits.pids),
             "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--entrypoint", "bash",
             image, "/tmp/run.sh"], timeout=120)
        with tempfile.TemporaryDirectory() as d:
            for fname, text in (("run.sh", _wrapper(bool(patch.strip()))), ("eval.sh", eval_script),
                                ("patch.diff", patch)):
                (Path(d) / fname).write_bytes(text.encode("utf-8"))
                _sh(["docker", "cp", str(Path(d) / fname), f"{name}:/tmp/{fname}"], timeout=120)
        try:
            p = _sh(["docker", "start", "-a", name], timeout=limits.timeout_s, check=False)
            log, timed_out = p.stdout + p.stderr, False
        except subprocess.TimeoutExpired as e:
            _sh(["docker", "kill", name], check=False)
            raw = (e.stdout or b"") if isinstance(e.stdout, bytes) else (e.stdout or "")
            log, timed_out = (raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw), True
        return SandboxResult(log, timed_out, time.time() - t0, image_id=image_id, docker_server=info, **meta)
    except (subprocess.SubprocessError, OSError) as e:
        err = (getattr(e, "stderr", "") or str(e))[:500]
        return SandboxResult("", False, time.time() - t0, image_id=image_id, docker_server=info,
                             error=f"sandbox failure: {err}", **meta)
    finally:
        _sh(["docker", "rm", "-f", name], check=False)
