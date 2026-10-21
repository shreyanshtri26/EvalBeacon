"""Redaction regression tests: values, keys, logs, spans and the judge prompt. All secrets here are fake."""
import json
import logging

import pytest
from test_instrumentor import _EXPORTER

from src.evalbeacon.core import Case, Output, run_eval
from src.evalbeacon.judge import JudgeReply
from src.evalbeacon.redact import SENSITIVE_KEYS, install_log_redaction, redact, redact_obj

OPENAI = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0"
GEMINI = "AIza" + "SyA1b2C3d4E5f6G7h8I9j0K1l2M3"
AWS = "AKIA" + "ABCDEFGHIJKLMNOP"
GH = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2"
JWT = "eyJhbGciOiJIUzI1NiJ9" + "." + "eyJzdWIiOiIxMjM0NTY3ODkwIn0" + "." + "dBjftJeZ4CVPmB92K27uhbUJU1p1r"
CARD = "4111 1111 1111 1111"  # standard test number, passes Luhn
SECRETS = [OPENAI, GEMINI, AWS, GH, JWT, "4111 1111 1111 1111", "123-45-6789", "alice@corp.io", "hunter2"]


@pytest.mark.parametrize("text", [
    f"key {OPENAI}", f"g={GEMINI}", f"aws {AWS}", f"gh {GH}", f"jwt {JWT}", f"card {CARD}", "ssn 123-45-6789",
    "mail alice@corp.io", "password=hunter2", 'json {"api_key": "hunter2"}', "token: 'hunter2'", "Bearer abcdefghijklmnop1234",
    "Authorization: hunter2",
])
def test_value_patterns(text):
    out = redact(text)
    for leaked in (OPENAI, GEMINI, AWS, GH, JWT, "4111", "123-45-6789", "alice@corp.io", "hunter2", "abcdefghijklmnop1234"):
        assert leaked not in out, (text, out)


def test_ordinary_numbers_and_text_are_left_alone():
    assert redact("Order 1234567890123 shipped in 3-5 business days, 12 month warranty") == \
        "Order 1234567890123 shipped in 3-5 business days, 12 month warranty"  # fails Luhn, so it is not a card


@pytest.mark.parametrize("key", ["password", "Password", "API_KEY", "Api-Key", "apiKey", "X-Auth-Token", "client_secret",
                                 "customerId", "Customer_ID", "account_number", "SSN", "authorization", "db_password"])
def test_sensitive_keys_lose_their_whole_value_regardless_of_casing(key):
    out = redact_obj({"ok": "fine", "nested": [{key: "value-that-has-no-pattern"}], key: {"inner": "x"}})
    assert out["ok"] == "fine"
    assert out[key] == "[REDACTED]" and out["nested"][0][key] == "[REDACTED]"
    assert "value-that-has-no-pattern" not in json.dumps(out)


def test_domain_specific_keys_via_environment(monkeypatch):
    assert redact_obj({"policy_number": "PN-99"})["policy_number"] == "PN-99"  # not sensitive by default
    monkeypatch.setenv("EVALBEACON_REDACT_KEYS", "Policy-Number, member_ref")
    out = redact_obj({"policy_number": "PN-99", "member_ref": "M1", "q": "hi"})
    assert out == {"policy_number": "[REDACTED]", "member_ref": "[REDACTED]", "q": "hi"}
    assert "customerid" in SENSITIVE_KEYS


def test_tuples_none_empty_values_and_cycles_do_not_crash():
    assert redact_obj({"password": None, "token": ""}) == {"password": None, "token": ""}  # nothing to hide
    assert redact_obj(("a@b.io", 5)) == ["[REDACTED_EMAIL]", 5]
    c: dict = {}
    c["self"] = c
    assert "TOO_DEEP" in json.dumps(redact_obj(c))  # bounded recursion instead of RecursionError


# ── results, spans, judge ───────────────────────────────────────────────────
def test_results_and_spans_never_contain_input_output_or_secrets():
    _EXPORTER.clear()
    question = "what is the SECRET-PROMPT-TEXT for me"
    cases = [Case("c1", {"question": question, "api_key": OPENAI, "customer_id": "C-77"}, {"must_contain": ["ok"]})]
    run = run_eval("r", lambda i: Output(f"ok COMPLETION-TEXT {OPENAI} {CARD}"), cases, "v")
    blob = json.dumps(run)
    for leaked in (OPENAI, "4111 1111", "C-77"):
        assert leaked not in blob
    spans = [s for s in _EXPORTER.get_finished_spans() if s.name == "evalbeacon.case"]
    attrs = json.dumps({k: str(v) for s in spans for k, v in s.attributes.items()})
    assert "SECRET-PROMPT-TEXT" not in attrs and "COMPLETION-TEXT" not in attrs and OPENAI not in attrs
    assert not any(e.attributes and "SECRET" in json.dumps(dict(e.attributes)) for s in spans for e in s.events)


def test_judge_prompt_is_redacted_even_for_domain_text():
    seen = []
    case = Case("j", {"question": "q"}, {"judge": {"rubric": "polite"}})
    run_eval("r", lambda i: Output(f"mail alice@corp.io card {CARD} {GEMINI}"), [case], "v",
             judge=lambda p: seen.append(p) or JudgeReply('{"score": 5}'))
    assert seen and all(x not in seen[0] for x in ("alice@corp.io", "4111", GEMINI))


def test_exception_text_in_failure_reasons_is_redacted():
    def agent(_):
        raise RuntimeError(f"401 Incorrect API key provided: {OPENAI}; contact alice@corp.io")

    run = run_eval("r", agent, [Case("c", {}, {})], "v")
    assert OPENAI not in json.dumps(run) and "alice@corp.io" not in json.dumps(run)
    assert OPENAI not in run["failure_summary"][0]["reason"]


# ── logging ─────────────────────────────────────────────────────────────────
def test_log_messages_extras_and_tracebacks_are_redacted(caplog):
    install_log_redaction()
    install_log_redaction()  # idempotent
    log = logging.getLogger("redaction.test")
    with caplog.at_level(logging.INFO, logger="redaction.test"):
        try:
            raise ValueError(f"upstream said: bad key {OPENAI}")
        except ValueError as exc:
            log.error("failed with %s for %s", exc, "alice@corp.io", exc_info=True, extra={"error": f"raw {GEMINI}", "n": 3})
    rec = caplog.records[-1]
    everything = caplog.text + str(rec.__dict__)
    for leaked in (OPENAI, GEMINI, "alice@corp.io"):
        assert leaked not in everything
    assert rec.exc_info is None and "ValueError" in rec.exc_text  # the traceback is kept, redacted, in exc_text
    assert rec.n == 3  # non-string extras are untouched


def test_log_record_exporters_cannot_read_the_raw_exception():
    """The OpenTelemetry logging handler reads record.exc_info; it must see nothing sensitive."""
    install_log_redaction()
    seen = []

    class Spy(logging.Handler):
        def emit(self, record):
            seen.append((record.exc_info, record.getMessage(), record.__dict__.get("exc_text")))

    log = logging.getLogger("redaction.spy")
    log.addHandler(Spy())
    log.setLevel(logging.INFO)
    try:
        raise RuntimeError(OPENAI)
    except RuntimeError:
        log.exception("boom %s", GEMINI)
    exc_info, msg, text = seen[0]
    assert exc_info is None and GEMINI not in msg and OPENAI not in (text or "")


def test_redaction_keeps_record_args_so_uvicorn_access_logs_still_format():
    """Regression: clearing record.args made uvicorn's AccessFormatter raise 'not enough values to unpack'."""
    from uvicorn.logging import AccessFormatter

    install_log_redaction()
    rec = logging.getLogRecordFactory()("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
                                        ("127.0.0.1:5000", "GET", f"/x?api_key={OPENAI}&q=1", "1.1", 200), None)
    assert isinstance(rec.args, tuple) and len(rec.args) == 5  # structure preserved
    line = AccessFormatter(fmt="%(levelprefix)s %(client_addr)s - \"%(request_line)s\" %(status_code)s").format(rec)
    assert OPENAI not in line and "GET" in line and "200" in line


def test_secret_inside_a_non_string_arg_is_still_redacted():
    install_log_redaction()
    rec = logging.getLogRecordFactory()("n", logging.ERROR, "", 0, "failed: %s", (RuntimeError(f"bad key {OPENAI}"),), None)
    assert OPENAI not in rec.getMessage() and "failed:" in rec.getMessage()
