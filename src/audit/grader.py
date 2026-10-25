"""Reference grader: the upstream swebench grading functions applied to a sandbox log (pinned version)."""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path

from swebench.harness.grading import get_eval_report
from swebench.harness.utils import make_test_spec

GRADER_ID = "swebench-grading"


def grader_version() -> str:
    return f"{GRADER_ID}=={version('swebench')}"


@dataclass
class Verdict:
    resolved: bool
    patch_applied: bool
    infra_failure: bool
    f2p_passed: int = 0
    f2p_total: int = 0
    p2p_passed: int = 0
    p2p_total: int = 0
    failed_tests: list[str] = field(default_factory=list)
    reason: str = ""
    grader: str = ""

    @property
    def label(self) -> str:
        return "RESOLVED" if self.resolved else "UNRESOLVED"


def grade(task: dict, restricted: dict, log: str, patch: str) -> Verdict:
    """Grade one sandbox log. `task` is the public row, `restricted` the grader-only row for the same instance."""
    row = {**task, **restricted}
    spec = make_test_spec(row)
    with tempfile.TemporaryDirectory() as d:
        fp = Path(d) / "test_output.txt"
        fp.write_text(log, encoding="utf-8")
        rep = get_eval_report(spec, {"instance_id": task["instance_id"], "model_name_or_path": "audit",
                                     "model_patch": patch}, str(fp), include_tests_status=True)[task["instance_id"]]
    ts = rep.get("tests_status", {})
    f2p, p2p = ts.get("FAIL_TO_PASS", {}), ts.get("PASS_TO_PASS", {})
    nf, np_ = len(f2p.get("success", [])), len(p2p.get("success", []))
    tf, tp = nf + len(f2p.get("failure", [])), np_ + len(p2p.get("failure", []))
    failed = [*f2p.get("failure", []), *p2p.get("failure", [])]
    reason = ""
    if not rep["patch_successfully_applied"]:
        reason = "no parseable test output (patch did not apply, timeout, or environment failure)"
    elif not rep["resolved"]:
        reason = f"{len(f2p.get('failure', []))} FAIL_TO_PASS and {len(p2p.get('failure', []))} PASS_TO_PASS failing"
    return Verdict(rep["resolved"], rep["patch_successfully_applied"], bool(rep.get("infra_failure")),
                   nf, tf, np_, tp, failed[:50], reason, grader_version())
