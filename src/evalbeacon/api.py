"""Dashboard + report endpoints. The UI is a single static file that consumes /api/v1/eval/report."""
import logging
import re
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, SecretStr, field_validator

from src.evalbeacon import history
from src.evalbeacon.cli import build_report
from src.evalbeacon.livecheck import (
    LIVE_PROVIDERS,
    MODEL_PATTERN,
    MODELS,
    LiveCheckError,
    RateLimiter,
    list_models,
    run_live_check,
    valid_model,
)

router = APIRouter()
logger = logging.getLogger(__name__)
limiter = RateLimiter(max_calls=5, window_s=60)
NO_STORE = {"Cache-Control": "no-store", "Pragma": "no-cache"}
ROOT = Path(__file__).resolve().parents[2]  # works from any working directory (e.g. a serverless runtime)
UI = Path(__file__).parent / "ui" / "index.html"
DATASETS = {"core": str(ROOT / "datasets/support.jsonl"), "extended": str(ROOT / "datasets/support_extended.jsonl")}
POLICY = str(ROOT / "configs/gate.json")
SCENARIOS = ("candidate-bad", "candidate-good", "candidate-flaky")


@router.get("/api/v1/eval/report")
def report(candidate: str = Query("candidate-bad", pattern="^candidate-(bad|good|flaky)$"),
           dataset: str = Query("core", pattern="^(core|extended)$")):
    """Run the offline demo (deterministic, no provider key) and return baseline, candidate and gate."""
    try:
        return build_report(DATASETS[dataset], candidate, POLICY)
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/api/v1/eval/history")
def runs(limit: int = Query(20, ge=1, le=200)):
    """Saved runs (newest first). With none saved (a fresh or read-only deploy) a labelled sample is returned."""
    rows = history.list_runs(Path("results/history"))
    if rows:
        return [{**r, "sample": False} for r in rows[:limit]]
    sample = []
    for scenario in SCENARIOS:  # mock data: the three demo scenarios against the extended dataset
        rep = build_report(DATASETS["extended"], scenario, POLICY)
        sample.append(history.summarize(rep["candidate"]) | {"sample": True, "gate": rep["gate"]["decision"]})
    return sample[:limit]


@router.get("/logo.png", include_in_schema=False)
def logo():
    return FileResponse(ROOT / "assets" / "logo.png", media_type="image/png")


@router.get("/logo-64.png", include_in_schema=False)
def logo_small():
    return FileResponse(ROOT / "assets" / "logo-64.png", media_type="image/png")


@router.get("/ui", include_in_schema=False)
def ui():
    return FileResponse(UI)


@router.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/ui")


def _validate_key(v: SecretStr) -> SecretStr:
    raw = v.get_secret_value().strip()
    if not re.fullmatch(r"[\x21-\x7e]{8,400}", raw):  # printable ASCII, no spaces or control characters
        raise ValueError("the key must be one token of printable characters with no spaces")
    return SecretStr(raw)


class LiveCheckRequest(BaseModel):
    provider: Literal["openai", "gemini", "deepseek", "anthropic"]
    api_key: SecretStr = Field(min_length=8, max_length=400)  # SecretStr: never rendered by repr, str() or logging
    model: str | None = Field(None, pattern=MODEL_PATTERN)

    @field_validator("model")
    @classmethod
    def _clean_model(cls, v: str | None) -> str | None:
        if v is not None and not valid_model(v):
            raise ValueError("invalid model name")
        return v

    @field_validator("api_key")
    @classmethod
    def _clean_key(cls, v: SecretStr) -> SecretStr:
        return _validate_key(v)


def _client_id(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")


@router.post("/api/v1/eval/live-check")
def live_check(body: LiveCheckRequest, request: Request):
    """Run the release gate against a real provider with a key supplied in the request body.

    The key is used for this request only and is never stored, logged or returned. See src/evalbeacon/livecheck.py.
    """
    wait = limiter.check(_client_id(request))
    if wait:
        return JSONResponse(status_code=429, headers={**NO_STORE, "Retry-After": str(int(wait))},
                            content={"ok": False, "stage": "limit", "error": {"code": "too_many_requests",
                                     "message": f"Too many live checks from this client. Try again in {int(wait)}s."}})
    key = body.api_key.get_secret_value()
    try:
        report = run_live_check(body.provider, key, body.model, DATASETS["core"], POLICY)
    except LiveCheckError as e:
        logger.info("live check refused: provider=%s code=%s", body.provider, e.code)  # never the key or provider text
        return JSONResponse(status_code=e.status, headers=NO_STORE,
                            content={"ok": False, "stage": "preflight", "error": {"code": e.code, "message": e.message}})
    except (OSError, ValueError):
        logger.exception("live check failed for provider=%s", body.provider)
        return JSONResponse(status_code=500, headers=NO_STORE, content={"ok": False, "stage": "run",
                            "error": {"code": "internal_error", "message": "The live check could not be completed."}})
    logger.info("live check done: provider=%s decision=%s", body.provider, report["gate"]["decision"])
    return JSONResponse(content=report, headers=NO_STORE)


class ModelsRequest(BaseModel):
    provider: Literal["openai", "gemini", "deepseek", "anthropic"]
    api_key: SecretStr = Field(min_length=8, max_length=400)

    @field_validator("api_key")
    @classmethod
    def _clean_key(cls, v: SecretStr) -> SecretStr:
        return _validate_key(v)  # same key rules as the live check


@router.post("/api/v1/eval/models")
def live_models(body: ModelsRequest, request: Request):
    """The chat models this key can use right now (cheapest first), straight from the provider's own model list.

    Read-only and free for the user. The key is used for this request only and never stored, logged or returned.
    """
    wait = limiter.check(_client_id(request))
    if wait:
        return JSONResponse(status_code=429, headers={**NO_STORE, "Retry-After": str(int(wait))},
                            content={"ok": False, "error": {"code": "too_many_requests", "message": f"Too many requests. Try again in {int(wait)}s."}})
    try:
        models = list_models(body.provider, body.api_key.get_secret_value())
    except LiveCheckError as e:
        logger.info("model list refused: provider=%s code=%s", body.provider, e.code)
        return JSONResponse(status_code=e.status, headers=NO_STORE, content={"ok": False, "error": {"code": e.code, "message": e.message}})
    return JSONResponse(headers=NO_STORE, content={"ok": True, "provider": body.provider, "models": models})


@router.get("/api/v1/eval/providers")
def providers():
    """Providers the live check supports, with their default (low-cost) model."""
    from src.collectors.providers import CHEAP_MODELS
    return [{"id": p, "default_model": CHEAP_MODELS[p], "models": MODELS[p]} for p in LIVE_PROVIDERS]


def install_error_handlers(app: FastAPI) -> None:
    """422 bodies without the rejected input: FastAPI's default echoes it back, which would return a pasted key."""
    @app.exception_handler(RequestValidationError)
    async def _validation(_request: Request, exc: RequestValidationError):
        errs = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errs}, headers=NO_STORE)
