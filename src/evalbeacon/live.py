"""Live-provider agents (OpenAI, DeepSeek, Gemini) via the shared instrumented client."""
from __future__ import annotations

from src.collectors.providers import chat_kwargs, default_model, make_client
from src.evalbeacon.core import Output
from src.evalbeacon.demo import _lookup
from src.evalbeacon.judge import JudgeReply

GROUNDED = "You are a support agent. Answer ONLY from the provided context, quoting its wording. If the context is empty say: I am not sure; please contact support."
UNGROUNDED = "You are a friendly support agent. Answer in a chatty, general way."


def make_judge(provider: str, model: str | None = None):
    """LLM judge backed by a provider; temperature 0, tiny output."""
    model = model or default_model(provider)
    client = make_client(provider, timeout=30, max_retries=1)

    def judge(prompt: str) -> JudgeReply:
        r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                           **chat_kwargs(provider, model, 150))
        u = r.usage
        return JudgeReply(r.choices[0].message.content or "", model,
                          u.prompt_tokens if u else None, u.completion_tokens if u else None)
    return judge


def make_embedder(provider: str, model: str = "text-embedding-3-small"):
    """Embeddings via an OpenAI-compatible /embeddings endpoint (OpenAI, custom)."""
    client = make_client(provider, timeout=30, max_retries=1)
    return lambda texts: [d.embedding for d in client.embeddings.create(model=model, input=texts).data]


def make_agent(provider: str, model: str | None, *, grounded: bool, max_retries: int = 1, api_key: str | None = None):
    """Return (agent, model). Baseline is grounded on retrieved context; candidate is not.

    Retry contract: the SDK client retries only transport-level failures (connection errors, 408/409/429/5xx)
    up to `max_retries` times; every retry is the same side-effect-free chat request. Tool side effects live in
    agent code that runs after a response, and the runner never re-invokes an agent. Use max_retries=0 for strict
    one-request-per-case accounting."""
    model = model or default_model(provider)
    client = make_client(provider, key=api_key, timeout=30, max_retries=max_retries)

    def agent(inp: dict) -> Output:
        q = inp["question"]
        ctx = (_lookup(q) or "") if grounded else ""
        msgs = [{"role": "system", "content": GROUNDED if grounded else UNGROUNDED},
                {"role": "user", "content": f"Context: {ctx}\n\nQuestion: {q}" if grounded else q}]
        r = client.chat.completions.create(model=model, messages=msgs, **chat_kwargs(provider, model, 200))
        u = r.usage
        # retrieval (_lookup) is the agent's one tool step; only the grounded agent performs it
        return Output(r.choices[0].message.content or "", u.prompt_tokens if u else None,
                      u.completion_tokens if u else None, ["kb_search"] if grounded else [])
    return agent, model
