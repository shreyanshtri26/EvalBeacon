"""Public leaderboard submissions (SWE-bench/experiments): metadata, original verdicts, trajectories, leakage scan.

Only the mini-SWE-agent trajectory layout (trajectory_format 'mini-swe-agent-*') is parsed; other layouts are reported
as unsupported rather than guessed."""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = "SWE-bench/experiments"
RAW = f"https://raw.githubusercontent.com/{REPO}/main/evaluation"
API = f"https://api.github.com/repos/{REPO}/contents/evaluation"
S3 = "https://swe-bench-submissions.s3.amazonaws.com"
CACHE = Path("data/audit/submissions")

# Commands that can read repository history or other refs, i.e. could reach commits newer than the base commit.
# Plain `git diff` / `status` / `checkout -- path` are excluded: they show the agent's own working tree, not history.
# Plain `git diff`/`status`/`checkout -- path` are excluded: they show the agent's own working tree, not history.
_HISTORY = re.compile(
    r"\bgit\s+(?:-C\s+\S+\s+)?(?:--no-pager\s+)?(log|show|reflog|rev-list|rev-parse|cat-file|fsck|fetch|pull|"
    r"ls-remote|remote|tag|branch|show-ref|for-each-ref)\b[^\n|;&]*")
_BROAD = re.compile(r"--all\b|--branches|--tags|--remotes|\bfsck\b|\bfetch\b|\bls-remote\b|\breflog\b|show-ref|"
                    r"for-each-ref|cat-file|rev-list")


def _get(url: str, binary: bool = False, timeout: float = 60.0):
    req = urllib.request.Request(url, headers={"User-Agent": "swebench-audit",
                                               **({"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"}
                                                  if "GITHUB_TOKEN" in os.environ and "api.github" in url else {})})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            break
        except urllib.error.HTTPError as e:  # 4xx is an answer (e.g. missing file); only 429/5xx are transient
            if e.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 3:
                raise
        time.sleep(2 ** attempt)
    return data if binary else data.decode("utf-8")


def list_entries(split: str = "verified") -> list[str]:
    return sorted(e["name"] for e in json.loads(_get(f"{API}/{split}")) if e["type"] == "dir")


@dataclass
class Entry:
    name: str
    split: str
    meta: dict
    verdicts: dict[str, bool]  # original grader outcome per instance id
    verdict_source: str

    @property
    def trajs_url(self) -> str | None:
        t = (self.meta.get("assets") or {}).get("trajs")
        return t if isinstance(t, str) else None

    def comparability(self) -> dict:
        """Protocol fields recorded for the comparability matrix. None means the submission does not state it."""
        info, tags = self.meta.get("info") or {}, self.meta.get("tags") or {}
        sysd = tags.get("system") or {}
        return {"entry": self.name, "agent": tags.get("agent"), "agent_version": info.get("mini-swe-agent_version"),
                "model": ",".join(tags["model"]) if isinstance(tags.get("model"), list) else tags.get("model"),
                "reasoning_effort": tags.get("reasoning_effort"), "attempts": sysd.get("attempts"),
                "open_source_system": tags.get("os_system"), "reported_resolved_pct": info.get("resolved"),
                "reported_cost_usd": info.get("cost"), "instance_calls": info.get("instance_calls"),
                "checked": tags.get("checked"), "trajs_available": self.trajs_url is not None,
                "logs_available": (self.meta.get("assets") or {}).get("logs") not in (None, "null"),
                "verdict_source": self.verdict_source, "verdict_count": len(self.verdicts)}


def load_entry(name: str, split: str = "verified", cache: Path = CACHE) -> Entry:
    d = cache / split / name
    d.mkdir(parents=True, exist_ok=True)

    def cached(rel: str) -> str | None:
        f = d / rel.replace("/", "__")
        if not f.exists():
            try:
                f.write_text(_get(f"{RAW}/{split}/{name}/{rel}"), encoding="utf-8")
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                raise
        return f.read_text(encoding="utf-8")

    meta = yaml.safe_load(cached("metadata.yaml") or "{}") or {}
    verdicts: dict[str, bool] = {}
    src = "none"
    if (txt := cached("per_instance_details.json")) is not None:
        verdicts = {k: bool(v["resolved"]) for k, v in json.loads(txt).items()}
        src = "per_instance_details.json"
    elif (txt := cached("results/results.json")) is not None:
        res = json.loads(txt)
        resolved = set(res.get("resolved", []))
        every = resolved | set(res.get("no_generation", [])) | set(res.get("no_logs", [])) | set(res.get("unresolved", []))
        verdicts = {i: i in resolved for i in every} if every else {}
        src = "results/results.json"
    return Entry(name, split, meta, verdicts, src)


@dataclass
class Traj:
    instance_id: str
    patch: str | None
    exit_status: str | None
    commands: list[tuple[str, str]] = field(default_factory=list)  # (command, raw output)
    fmt: str = ""


def traj_url(e: Entry, instance_id: str) -> str | None:
    base = e.trajs_url
    if not base or not base.startswith("s3://swe-bench-submissions/"):
        return None
    return f"{S3}/{base.removeprefix('s3://swe-bench-submissions/')}/{instance_id}/{instance_id}.traj.json"


_BASH_BLOCK = re.compile(r"```bash\n(.*?)\n```", re.S)


def _text(content) -> str:
    return content if isinstance(content, str) else "".join(c.get("text", "") for c in content or [])


def _parse_legacy(raw: dict) -> Traj:
    """mini-SWE-agent <= 1.x: the command is a ```bash block in the assistant text, the output is the next user turn."""
    msgs = raw.get("messages", [])
    cmds: list[tuple[str, str]] = []
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant" or not (blk := _BASH_BLOCK.search(_text(m.get("content")))):
            continue
        nxt = msgs[i + 1] if i + 1 < len(msgs) and msgs[i + 1].get("role") == "user" else {}
        cmds.append((blk.group(1), _text(nxt.get("content"))))
    info = raw.get("info", {})
    return Traj(raw.get("instance_id", ""), info.get("submission"), info.get("exit_status"), cmds, "mini-swe-agent-legacy")


def parse_traj(raw: dict) -> Traj:
    fmt = str(raw.get("trajectory_format", ""))
    if not fmt and "submission" in raw.get("info", {}) and any(
            m.get("role") == "assistant" and "```bash" in _text(m.get("content")) for m in raw.get("messages", [])):
        return _parse_legacy(raw)
    if not fmt.startswith("mini-swe-agent"):
        raise ValueError(f"unsupported trajectory format {fmt!r}")
    msgs = raw.get("messages", [])
    outputs = {m["tool_call_id"]: (m.get("extra") or {}).get("raw_output", m.get("content", ""))
               for m in msgs if m.get("role") == "tool" and m.get("tool_call_id")}
    cmds: list[tuple[str, str]] = []
    for m in msgs:
        for tc in (m.get("tool_calls") or []) if m.get("role") == "assistant" else []:
            try:
                cmd = json.loads(tc["function"]["arguments"]).get("command", "")
            except (ValueError, KeyError, TypeError):
                continue
            cmds.append((cmd, outputs.get(tc.get("id"), "")))
    info = raw.get("info", {})
    return Traj(raw.get("instance_id", ""), info.get("submission"), info.get("exit_status"), cmds, fmt)


def fetch_traj(e: Entry, instance_id: str, cache: Path = CACHE, keep: bool = True) -> Traj | None:
    """Download and parse one trajectory. keep=False streams it (trajectories are ~0.5 MB each; a study needs thousands)."""
    f = cache / e.split / e.name / "trajs" / f"{instance_id}.traj.json"
    if f.exists():
        return parse_traj(json.loads(f.read_text(encoding="utf-8")))
    url = traj_url(e, instance_id)
    if url is None:
        return None
    try:
        data = _get(url, binary=True, timeout=120)
    except urllib.error.HTTPError:
        return None
    if keep:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)
    return parse_traj(json.loads(data))


def _added_lines(patch: str, min_len: int = 30) -> set[str]:
    """Distinctive added lines of a patch; short or boilerplate lines are ignored because they match anything."""
    out = set()
    for ln in patch.splitlines():
        if ln.startswith("+") and not ln.startswith("+++"):
            s = ln[1:].strip()
            if len(s) >= min_len and not s.startswith(("#", "//", '"""', "'''")):
                out.add(s)
    return out


def leak_scan(traj: Traj, gold_patch: str | None = None) -> dict:
    """Observable evidence only; none of this proves a result was obtained by cheating.

    history_commands: git commands that can read history or other refs (weakest signal; routine commands included).
    broad_history:    the subset that can reach newer commits (--all, fetch, reflog, cat-file, ...).
    gold_overlap:     distinctive gold-patch lines appearing in the OUTPUT of a history command (strongest signal; a
                      human must still confirm that the agent saw and used them). Needs the restricted gold patch."""
    hist, broad = [], []
    for i, (cmd, _) in enumerate(traj.commands):
        for m in _HISTORY.finditer(cmd):
            hist.append({"step": i, "command": m.group(0).strip()[:160]})
            if _BROAD.search(m.group(0)):
                broad.append({"step": i, "command": m.group(0).strip()[:160]})
    overlap: list[dict] = []
    if gold_patch:
        gold = _added_lines(gold_patch)
        for i, (cmd, out) in enumerate(traj.commands):
            if re.search(r"\bgit\s+(?:-C\s+\S+\s+)?(?:--no-pager\s+)?(log|show|cat-file|reflog)\b", cmd):
                shown = {ln[1:].strip() for ln in out.splitlines() if ln.startswith("+")}  # as an added diff line
                hit = sorted(gold & shown)
                if hit:
                    overlap.append({"step": i, "command": cmd[:160], "matched_lines": len(hit), "of_gold_lines": len(gold)})
    level = "gold_overlap" if overlap else "broad_history" if broad else "history_commands" if hist else "none"
    return {"level": level, "history_commands": hist, "broad_history": broad, "gold_overlap": overlap}
