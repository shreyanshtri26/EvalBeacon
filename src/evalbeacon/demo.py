"""Deterministic offline demo agents: a good baseline and a deliberately degraded candidate."""
from src.evalbeacon.core import Output

_KB = {
    "refund": "Our return policy allows refunds within 30 days of purchase with a receipt.",
    "shipping": "Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days.",
    "warranty": "All products include a 12 month warranty covering manufacturing defects.",
    "password": "Use the reset password link on the sign in page; we never ask for your password by email.",
    "invoice": "Invoices are emailed after checkout and available under Account > Billing.",
    "cancel": "You can cancel an order before it ships from Account > Orders.",
    "contact": "Contact support at help@example.com or via live chat from 9am to 6pm.",
    "exchange": "Exchanges follow the same return policy: 30 days, original packaging required.",
}


def _lookup(q: str) -> str | None:
    q = q.lower()
    return next((v for k, v in _KB.items() if k in q), None)


def baseline(inp: dict) -> Output:
    q = inp["question"]
    ans = _lookup(q) or "I am not sure; please contact support."
    return Output(ans, input_tokens=40 + len(q), output_tokens=len(ans) // 4, tool_calls=["kb_search"])


def candidate_bad(inp: dict) -> Output:
    """Regressed prompt: loses grounding on some topics, rambles, and skips the tool call."""
    q = inp["question"]
    ans = _lookup(q)
    if ans is None or len(q) % 2 == 0:
        ans = "Sure! Great question. Generally it depends, but we are happy to help in any way we can. " * 2
    return Output(ans, input_tokens=90 + len(q), output_tokens=len(ans) // 3, tool_calls=[])


def kb_search(question: str) -> str | None:
    """The agent's one tool: read-only, safe to record and replay."""
    return _lookup(question)


def toolbox_baseline(inp: dict, tools) -> Output:
    """Baseline logic expressed through a tool interface (Recorder or Replayer)."""
    q = inp["question"]
    ans = tools.call("kb_search", question=q) or "I am not sure; please contact support."
    return Output(ans, input_tokens=40 + len(q), output_tokens=len(ans) // 4, tool_calls=list(tools.called))


def candidate_flaky(inp: dict) -> Output:
    """Right answers, but the provider fails on a deterministic third of requests (rate limits)."""
    q = inp["question"]
    if sum(map(ord, q)) % 3 == 0:
        raise RuntimeError("429 rate limit exceeded (mock provider)")
    return baseline(inp)


AGENTS = {"baseline": baseline, "candidate-bad": candidate_bad, "candidate-good": baseline,
          "candidate-flaky": candidate_flaky}
