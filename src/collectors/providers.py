"""Supported providers. All three speak the OpenAI chat API, so one client covers them.

Keys come from the environment / .env (never logged). Env var names are matched case-insensitively
because the project's .env uses `Gemini_API_KEY` / `DeepSeek_API_KEY`.
"""
from __future__ import annotations

import os
import re

from dotenv import load_dotenv

PROVIDERS = {  # name: (base_url, key env var, default model)
    "openai": (None, "OPENAI_API_KEY", "gpt-4o-mini"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY", "deepseek-flash"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY", "gemini-2.5-flash-lite"),
    # Anthropic's OpenAI-SDK-compatible endpoint: fine for evaluation and comparison, but Anthropic documents it as a
    # compatibility layer, not the full Messages API (no prompt caching, extended thinking or similar features).
    "anthropic": ("https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY", "claude-haiku-5-5"),
}
CHEAP_MODELS = {p: v[2] for p, v in PROVIDERS.items()}  # defaults are the low-cost model of each provider
# Any other OpenAI-compatible endpoint (Ollama, vLLM, LiteLLM, Together...) via env:
#   EVALBEACON_CUSTOM_BASE_URL, EVALBEACON_CUSTOM_API_KEY, EVALBEACON_CUSTOM_MODEL
load_dotenv()
if os.getenv("EVALBEACON_CUSTOM_BASE_URL"):
    PROVIDERS["custom"] = (os.environ["EVALBEACON_CUSTOM_BASE_URL"], "EVALBEACON_CUSTOM_API_KEY",
                           os.getenv("EVALBEACON_CUSTOM_MODEL", "default"))
DEFAULT_PROVIDER = "openai"


def chat_kwargs(provider: str, model: str, max_tokens: int, reasoning_floor: int = 1024) -> dict:
    """Request parameters that the model family accepts.

    OpenAI's gpt-5.x and o-series are reasoning models: they reject `max_tokens` and a non-default `temperature`,
    and spend part of the budget on hidden reasoning, so they get `max_completion_tokens` with a floor."""
    m = model.lower()
    if provider == "openai" and (m.startswith("gpt-5") or re.match(r"o\d", m)):
        return {"max_completion_tokens": max(max_tokens, reasoning_floor)}
    return {"temperature": 0, "max_tokens": max_tokens}


def default_model(provider: str) -> str:
    return PROVIDERS[provider][2]


def api_key(provider: str) -> str:
    var = PROVIDERS[provider][1]
    load_dotenv()
    for k, v in os.environ.items():
        if k.upper() == var and v:
            return v
    raise ValueError(f"{var} is not set (add it to .env)")


def make_client(provider: str, *, key: str | None = None, **kwargs):
    """Instrumented OpenAI-compatible client for `provider`.

    `key` is an explicit API key (used by the paste-a-key web check and never stored); without it the key comes
    from the environment / .env."""
    from src.collectors.openai_collector import InstrumentedOpenAI

    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; choose from {sorted(PROVIDERS)}")
    return InstrumentedOpenAI(provider=provider, api_key=key or api_key(provider),
                              base_url=PROVIDERS[provider][0], **kwargs)
