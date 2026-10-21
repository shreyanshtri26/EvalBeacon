"""Redaction applied before any payload leaves the evaluation boundary.

Covers: results JSON, spans, API responses, payload stores, judge prompts and (via install_log_redaction) logs.

Two mechanisms, both best-effort and pattern-based (this is not a PII detector):
- value patterns: API keys and tokens, bearer/JWT tokens, key=value secrets, emails, US SSNs, payment card
  numbers (Luhn-checked to avoid redacting ordinary long numbers);
- sensitive keys: any dict key whose normalised name (lowercase, letters and digits only) is a known secret or
  identifier field has its whole value replaced, whatever the casing or separators (``Api-Key``, ``API_KEY``).

Add domain-specific fields with EVALBEACON_REDACT_KEYS=customer_ref,policy_number (comma separated) or by
extending SENSITIVE_KEYS / REDACTORS.
"""
from __future__ import annotations

import logging
import os
import re
import traceback
from typing import overload

# ponytail: pattern list, not a PII detector; extend REDACTORS or pass your own to redact()
REDACTORS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"sk-[A-Za-z0-9_\-]{12,}"), "[REDACTED_KEY]"),                       # OpenAI / DeepSeek style
    (re.compile(r"AIza[0-9A-Za-z_\-]{20,}"), "[REDACTED_KEY]"),                      # Google API keys
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_KEY]"),                         # AWS access key id
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "[REDACTED_KEY]"),               # GitHub tokens
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"), "[REDACTED_TOKEN]"),  # JWT
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}"), "Bearer [REDACTED_TOKEN]"),
    # key=value / "key": "value" pairs for secret-looking names, quoted or not
    (re.compile(r"""(?i)(["']?(?:api[_-]?key|secret|token|password|passwd|authorization)["']?\s*[=:]\s*["']?)[^\s"',}]+"""),
     r"\1[REDACTED]"),
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
]
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")

SENSITIVE_KEYS: set[str] = {
    "password", "passwd", "secret", "token", "apikey", "authorization", "auth", "accesstoken", "refreshtoken",
    "clientsecret", "privatekey", "sessionid", "cookie", "ssn", "cardnumber", "cvv", "cvc", "iban",
    "accountnumber", "customerid", "email", "phone", "dob", "dateofbirth",
}


def _norm(key: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _extra_keys() -> set[str]:
    return {_norm(k) for k in os.getenv("EVALBEACON_REDACT_KEYS", "").split(",") if k.strip()}


def _is_sensitive_key(key: object, extra: set[str]) -> bool:
    n = _norm(key)
    return n in SENSITIVE_KEYS or n in extra or n.endswith(("password", "secret", "apikey", "token"))


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for d in reversed(digits):
        n = int(d)
        if alt:
            n = n * 2 - 9 if n * 2 > 9 else n * 2
        total += n
        alt = not alt
    return total % 10 == 0


def _card(m: re.Match) -> str:
    digits = re.sub(r"\D", "", m.group(0))
    return "[REDACTED_CARD]" if 13 <= len(digits) <= 19 and _luhn(digits) else m.group(0)


@overload
def redact(text: str, extra: list[tuple[re.Pattern, str]] | None = None) -> str: ...
@overload
def redact(text: None, extra: list[tuple[re.Pattern, str]] | None = None) -> None: ...
def redact(text, extra=None):
    if text is None:
        return None
    for pat, repl in [*REDACTORS, *(extra or [])]:
        text = pat.sub(repl, text)
    return _CARD.sub(_card, text)


def redact_obj(obj, _extra: set[str] | None = None, _depth: int = 0):
    """Redact nested dicts/lists/tuples: sensitive keys lose their whole value, other strings are pattern-redacted."""
    extra = _extra_keys() if _extra is None else _extra
    if _depth > 50:  # cyclic or absurdly deep data: stop rather than recurse forever
        return "[REDACTED_TOO_DEEP]"
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: "[REDACTED]" if _is_sensitive_key(k, extra) and v not in (None, "") else redact_obj(v, extra, _depth + 1)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact_obj(v, extra, _depth + 1) for v in obj]
    return obj


# ── logging ─────────────────────────────────────────────────────────────────
_installed = False


def install_log_redaction() -> None:
    """Redact every log record at creation time, process-wide and idempotent.

    The rendered message is redacted, and exception tracebacks are pre-rendered and redacted into
    record.exc_text with record.exc_info cleared, so exporters that read exc_info (such as the OpenTelemetry
    logging handler) cannot ship the raw exception text. Trade-off: handlers that need the live exception
    object (for example some error trackers) will not receive it.
    """
    global _installed
    if _installed:
        return
    original = logging.getLogRecordFactory()

    def factory(*args, **kwargs):
        record = original(*args, **kwargs)
        try:
            # Keep the args structure: handlers such as uvicorn's access formatter unpack record.args by position.
            args = record.args
            if isinstance(args, tuple):
                record.args = tuple(redact(x) if isinstance(x, str) else x for x in args)
            elif isinstance(args, dict):
                record.args = {k: redact(x) if isinstance(x, str) else x for k, x in args.items()}
            if isinstance(record.msg, str):
                record.msg = redact(record.msg)
            final = record.getMessage()
            if redact(final) != final:  # a secret hidden in a non-string arg (e.g. an exception object): flatten
                record.msg, record.args = redact(final), ()
            if record.exc_info and record.exc_info[0] is not None:
                record.exc_text = redact("".join(traceback.format_exception(*record.exc_info)).rstrip())
                record.exc_info = None
        except Exception:  # noqa: BLE001 — logging must never raise
            record.msg, record.args = "[log message could not be redacted]", ()
        return record

    logging.setLogRecordFactory(factory)

    # `extra={...}` fields are attached by Logger.makeRecord *after* the factory runs, so redact them there too.
    original_make = logging.Logger.makeRecord

    def make_record(self, *args, **kwargs):
        record = original_make(self, *args, **kwargs)
        for k, v in list(record.__dict__.items()):
            if k not in _STD_ATTRS and isinstance(v, str):
                record.__dict__[k] = redact(v)
        return record

    logging.Logger.makeRecord = make_record  # type: ignore[method-assign]
    _installed = True


_STD_ATTRS = set(logging.LogRecord("n", 0, "p", 0, "m", (), None).__dict__) | {"message", "asctime", "exc_text"}
