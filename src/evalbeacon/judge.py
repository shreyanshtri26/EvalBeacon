"""Optional LLM-judge evaluator. Never the only check: deterministic evaluators still run and must pass.

The judge sees the *redacted* output only, runs at temperature 0, and its model, version, cost and
uncertainty are recorded with every verdict. A single sample is subjective; treat it as a signal.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from src.exporters.cost_estimator import CostEstimator

JUDGE_VERSION = "1"


@dataclass
class JudgeReply:
    text: str
    model: str = "unknown"
    input_tokens: int | None = None
    output_tokens: int | None = None


JudgeFn = Callable[[str], JudgeReply]


def build_prompt(question: str, output: str, rubric: str) -> str:
    return ("You are a strict grader. Score the ANSWER from 1 (bad) to 5 (excellent) against the RUBRIC.\n"
            'Reply with JSON only: {"score": <1-5>, "rationale": "<one sentence>"}.\n\n'
            f"QUESTION: {question}\nRUBRIC: {rubric}\nANSWER: {output}")


def parse_reply(text: str) -> tuple[int, str]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("judge reply contained no JSON object")
    d = json.loads(m.group(0))
    score = d["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 1 <= score <= 5:
        raise ValueError(f"judge score out of range: {score!r}")
    return int(round(score)), str(d.get("rationale", ""))[:300]


def judge_case(fn: JudgeFn, question: str, output: str, spec: dict) -> dict:
    reply = fn(build_prompt(question, output, spec["rubric"]))
    score, why = parse_reply(reply.text)
    cost = (CostEstimator.estimate(reply.model, reply.input_tokens, reply.output_tokens)
            if reply.input_tokens is not None and reply.output_tokens is not None else None)
    min_score = spec.get("min_score", 4)
    return {"score": score, "min_score": min_score, "passed": score >= min_score, "rationale": why,
            "judge_model": reply.model, "judge_version": JUDGE_VERSION, "cost_usd": cost,
            "uncertainty": "single sample at temperature 0; subjective, not a statistical measure"}
