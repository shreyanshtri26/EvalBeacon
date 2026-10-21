"""Replay with mocked tools. Recorded tool results are served back; a replay never calls a live tool.

Tools flagged as side-effecting (refunds, emails, payments) are refused by the Recorder entirely, so they
can neither run during recording nor appear in a cassette as if they had.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path


class SideEffectBlockedError(RuntimeError):
    pass


class ReplayMissError(RuntimeError):
    pass


def _key(tool: str, args: dict) -> str:
    return tool + json.dumps(args, sort_keys=True, default=str)


class Cassette:
    def __init__(self, entries: list[dict] | None = None):
        self.entries = entries or []

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"version": 1, "entries": self.entries}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Cassette:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("version") != 1:
            raise ValueError(f"{path}: unsupported cassette version {d.get('version')!r}")
        return cls(d["entries"])


class Recorder:
    """Calls real read-only tools and records (tool, args, result)."""

    def __init__(self, tools: dict[str, Callable], cassette: Cassette, side_effects: set[str] | frozenset[str] = frozenset()):
        self.tools, self.cassette, self.side_effects = tools, cassette, set(side_effects)
        self.called: list[str] = []

    def call(self, name: str, **args):
        if name in self.side_effects:
            raise SideEffectBlockedError(f"tool {name!r} has side effects and is never executed during evaluation")
        result = self.tools[name](**args)
        self.cassette.entries.append({"tool": name, "args": args, "result": result})
        self.called.append(name)
        return result


class Replayer:
    """Serves recorded results. An unrecorded call raises ReplayMissError; there is no fall-through to live tools."""

    def __init__(self, cassette: Cassette):
        self._by_key = {_key(e["tool"], e["args"]): e["result"] for e in cassette.entries}
        self.called: list[str] = []

    def call(self, name: str, **args):
        k = _key(name, args)
        if k not in self._by_key:
            raise ReplayMissError(f"no recorded result for {name}({args}); re-record the cassette")
        self.called.append(name)
        return self._by_key[k]
