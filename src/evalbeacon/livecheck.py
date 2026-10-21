"""Paste-a-key live check: run the baseline-vs-candidate gate against a real provider with a key the user supplies.

Security contract (each point is covered by tests/test_livecheck.py):
- The key arrives only in a POST body, is held as a SecretStr, and is used for this request only.
- It is never written to disk, history, payload stores or logs, never put in a URL, and never echoed back: the
  exact key string is scrubbed from the whole response before it is returned.
- Only whitelisted providers with their fixed base URLs are reachable, so a caller cannot point the server at an
  arbitrary host (no SSRF) or reuse it as a generic proxy.
- Spend and load are bounded: cheap default models, at most MAX_CASES cases, no retries, short timeouts, and a
  per-client rate limit.
- A one-token preflight runs first so a bad key, no credit or a rate limit gives a clear, safe message before
  any real cases are spent. Messages are fixed hints; provider error text is never forwarded.
"""
from __future__ import annotations

import re
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.collectors.providers import CHEAP_MODELS, PROVIDERS, chat_kwargs, make_client
from src.evalbeacon.core import load_dataset

LIVE_PROVIDERS = ("openai", "gemini", "deepseek", "anthropic")  # custom endpoints are deliberately not reachable from the web
MODEL_PATTERN = r"^[A-Za-z0-9._:/-]{1,100}$"  # also what a custom model name must match
# Suggested models, cheapest first. The first entry is each provider's default; any other name can be typed in.
# Price is not guessed here: costs for models missing from the price table are flagged as FALLBACK in the report.
# Verified against each provider's own model list (OpenAI, Gemini, DeepSeek); Anthropic's ids come from its current
# model family and could not be listed without a key. "Load models from my key" in the UI replaces this list with
# exactly what the pasted key can use, so the dropdown never has to be trusted blindly.
MODELS = {
    "openai": ["gpt-4o-mini", "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4o"],
    "gemini": ["gemini-2.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-2.5-flash", "gemini-3.5-flash"],
    "deepseek": ["deepseek-flash", "deepseek-v4-pro"],
    "anthropic": ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"],
}


def valid_model(name: str) -> bool:
    """Charset check plus no path-like tricks (`..`, leading or trailing `/`); the name is only ever sent to the provider."""
    return bool(re.fullmatch(MODEL_PATTERN, name)) and ".." not in name and not name.startswith("/") and not name.endswith("/")


MAX_CASES = 10
WORKERS = 5
CASE_TIMEOUT_S = 20.0
PREFLIGHT_TIMEOUT_S = 15.0
REDACTED = "[REDACTED_KEY]"
NAMES = {"openai": "OpenAI", "gemini": "Gemini", "deepseek": "DeepSeek", "anthropic": "Anthropic"}


class LiveCheckError(Exception):
    """A failure with a safe, user-facing explanation. Never carries provider text or the key."""

    def __init__(self, status: int, code: str, message: str, error_type: str = ""):
        super().__init__(message)
        self.status, self.code, self.message, self.error_type = status, code, message, error_type


def classify(exc: BaseException, provider: str) -> LiveCheckError:
    """Map a provider exception to a fixed hint. Only the exception class, HTTP status and error code are inspected."""
    name = type(exc).__name__
    provider = NAMES.get(provider, provider)  # display name in user-facing text
    status = getattr(exc, "status_code", None)
    code = str(getattr(exc, "code", "") or "").lower()
    text = str(exc).lower()
    if code in ("insufficient_quota", "billing_hard_limit_reached") or status == 402 or any(
            w in text for w in ("insufficient balance", "insufficient_quota", "credit balance", "exceeded your current quota")):
        return LiveCheckError(402, "no_credit", f"{provider} accepted the key but the account has no credit or quota left. "
                              "Add credit or use another key.", name)
    # Google reports a bad key as HTTP 400 ("API key not valid" / API_KEY_INVALID) rather than 401
    bad_key_400 = status == 400 and any(w in text for w in ("api key not valid", "api_key_invalid", "invalid api key", "api key expired", "valid api key"))
    if name in ("AuthenticationError", "PermissionDeniedError") or status in (401, 403) or bad_key_400:
        return LiveCheckError(401, "invalid_key", f"{provider} rejected this key. Check that you pasted the complete key "
                              "for this provider and that it has not been revoked.", name)
    if status == 429 and ("quota" in text or "resource_exhausted" in text):
        # e.g. Google: "Quota exceeded ... limit: 0" means the project has no free-tier quota for this model, which waiting will not fix
        return LiveCheckError(429, "quota_exceeded", f"{provider} reports the quota for this key or model is exhausted (some projects have "
                              "a limit of 0 for a model). Enable billing or quota in the provider console, try a different model, or wait a minute.", name)
    if name == "RateLimitError" or status == 429:
        return LiveCheckError(429, "rate_limited", f"{provider} is rate-limiting this key. Wait a moment and try again.", name)
    if name == "NotFoundError" or status == 404:
        return LiveCheckError(404, "model_not_found", f"{provider} does not know that model, or this key cannot use it. "
                              "Leave the model empty to use the default.", name)
    if name in ("APITimeoutError", "APIConnectionError") or isinstance(exc, (TimeoutError, ConnectionError)):
        return LiveCheckError(504, "unreachable", f"Could not reach {provider} in time. Try again shortly.", name)
    return LiveCheckError(502, "provider_error", f"{provider} returned an error ({name}"
                          + (f", status {status}" if isinstance(status, int) else "") + ").", name)


def preflight(provider: str, key: str, model: str) -> None:
    """One 1-token request: proves the key, the model and the account work before the real run spends anything."""
    try:
        client = make_client(provider, key=key, timeout=PREFLIGHT_TIMEOUT_S, max_retries=0)
        client.chat.completions.create(model=model, messages=[{"role": "user", "content": "ping"}],
                                       **chat_kwargs(provider, model, 1, reasoning_floor=16))
    except LiveCheckError:
        raise
    except Exception as exc:  # noqa: BLE001 — classified into a safe message
        raise classify(exc, provider) from None  # `from None`: never chain the original exception (it may quote the key)


_NOT_CHAT = re.compile(r"embed|tts|transcribe|whisper|image|dall-e|audio|realtime|moderation|sora|search|live|veo|lyria|aqa|imagen|"
                      r"robotics|computer-use|native|translate|codex|instruct|davinci|babbage|antigravity|research|omni|nano-banana|-latest$|"
                      r"-\d{4}-\d{2}-\d{2}$|-\d{4}$", re.I)


def _cost_rank(model_id: str) -> int:
    m = model_id.lower()
    return 0 if any(w in m for w in ("nano", "lite", "flash-lite")) else 1 if any(w in m for w in ("mini", "flash", "haiku")) else 2


def list_models(provider: str, key: str) -> list[str]:
    """Chat-capable models this key can use, cheapest first. Read-only and free; also proves the key works."""
    if provider not in LIVE_PROVIDERS:
        raise LiveCheckError(422, "unsupported_provider", f"provider must be one of {', '.join(LIVE_PROVIDERS)}")
    try:
        client = make_client(provider, key=key, timeout=PREFLIGHT_TIMEOUT_S, max_retries=0)
        ids = [m.id.removeprefix("models/") for m in client.models.list().data]
    except Exception as exc:  # noqa: BLE001
        raise classify(exc, provider) from None
    usable = sorted({i for i in ids if not _NOT_CHAT.search(i) and valid_model(i)}, key=lambda i: (_cost_rank(i), i))
    return usable[:40]


def scrub(obj: Any, secret: str) -> Any:
    """Replace every occurrence of the exact key (and its URL/JSON-escaped forms) anywhere in a JSON-like structure."""
    forms = {secret, secret.replace('"', '\\"')}
    if isinstance(obj, str):
        for f in forms:
            obj = obj.replace(f, REDACTED)
        return obj
    if isinstance(obj, dict):
        return {scrub(k, secret): scrub(v, secret) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub(v, secret) for v in obj]
    return obj


def run_live_check(provider: str, key: str, model: str | None, dataset: str, policy: str) -> dict:
    """Preflight, then baseline (grounded) vs candidate (ungrounded) on at most MAX_CASES cases. Returns a report dict."""
    from src.evalbeacon.cli import build_live_report

    if provider not in LIVE_PROVIDERS or provider not in PROVIDERS:
        raise LiveCheckError(422, "unsupported_provider", f"provider must be one of {', '.join(LIVE_PROVIDERS)}")
    if model is not None and not valid_model(model):
        raise LiveCheckError(422, "invalid_model", "The model name may only contain letters, digits and . _ : / - (no '..', no leading or trailing '/')")
    model = model or CHEAP_MODELS[provider]
    preflight(provider, key, model)
    report = build_live_report(dataset, provider, model, policy, workers=WORKERS, timeout=CASE_TIMEOUT_S, max_retries=0,
                               api_key=key, max_cases=MAX_CASES)
    return {**scrub(report, key), "ok": True, "provider": provider, "model": model,
            "cases": report["candidate"]["summary"]["cases"], "live": True}


def dataset_size(dataset: str) -> int:
    return min(len(load_dataset(Path(dataset))[0]), MAX_CASES)


class RateLimiter:
    """Best-effort sliding window per client. Serverless instances do not share memory, so this bounds bursts
    on one instance rather than guaranteeing a global limit."""

    def __init__(self, max_calls: int = 5, window_s: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.max_calls, self.window_s, self.clock = max_calls, window_s, clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, client: str) -> float:
        """Return 0 when allowed (and record the hit), otherwise seconds until the client may retry."""
        now = self.clock()
        with self._lock:
            q = self._hits[client]
            while q and now - q[0] > self.window_s:
                q.popleft()
            if len(q) >= self.max_calls:
                return max(1.0, self.window_s - (now - q[0]))
            q.append(now)
            if len(self._hits) > 10_000:  # bound memory under a flood of distinct clients
                self._hits.clear()
            return 0.0
