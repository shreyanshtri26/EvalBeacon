"""Module-level agents so they can be pickled for process isolation (and shared by tests)."""
import time
from pathlib import Path

from src.evalbeacon.core import Output, cancelled


def echo(inp: dict) -> Output:
    return Output(f"echo:{inp.get('q', '')}")


def boom(inp: dict) -> Output:
    raise ValueError("tool refused by policy")


def write_after_sleep(inp: dict) -> Output:
    """Side effect (a file) that happens only if the agent is allowed to finish."""
    time.sleep(inp["sleep"])
    Path(inp["path"]).write_text("side effect happened")
    return Output("done")


def cooperative(inp: dict) -> Output:
    """Polls cancelled() like a well-behaved long-running agent."""
    for _ in range(200):
        if cancelled():
            Path(inp["path"]).write_text("stopped cooperatively")
            return Output("cancelled")
        time.sleep(0.02)
    Path(inp["path"]).write_text("ran to completion")
    return Output("done")
