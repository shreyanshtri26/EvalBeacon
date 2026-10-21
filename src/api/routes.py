"""
routes.py
=========
Instrumented FastAPI route handlers demonstrating observability patterns.

Contract notes
- Inputs are validated (non-empty, bounded sizes, safe provider/model names); invalid bodies return 422.
- Unknown provider or missing API key returns 400; any upstream provider failure returns 502 with a
  *sanitised* message (error class, HTTP status and error code only). The raw provider message is never
  returned to clients; it is logged server-side after redaction.
- Handlers are plain `def` because the provider client is blocking; FastAPI runs them in its thread pool
  so a slow provider cannot stall the event loop.
- Token usage and cost are `null` when the provider reports no usage (unknown is not zero).
- User text is never attached to spans; only its length and a short hash are.
"""

import hashlib
import logging
import re
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.collectors.providers import DEFAULT_PROVIDER, PROVIDERS, default_model, make_client
from src.exporters.cost_estimator import CostEstimator

logger = logging.getLogger(__name__)
router = APIRouter()

NAME = r"^[A-Za-z0-9._:/-]{1,100}$"  # provider and model identifiers


def _model_for(provider: str) -> str:
    if provider not in PROVIDERS:
        raise HTTPException(status_code=400, detail=f"unknown provider {provider!r}; choose from {sorted(PROVIDERS)}")
    return default_model(provider)


def _provider_error(exc: Exception) -> HTTPException:
    """502 with a message that cannot carry credentials or provider-internal text."""
    parts = [type(exc).__name__]
    status, code = getattr(exc, "status_code", None), getattr(exc, "code", None)
    if isinstance(status, int):
        parts.append(f"status {status}")
    if isinstance(code, str) and re.fullmatch(r"[a-z0-9_.-]{1,64}", code):
        parts.append(f"code {code}")
    return HTTPException(status_code=502, detail=f"LLM provider error ({', '.join(parts)})")


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


# ─── Request / Response Models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    provider: str = Field(DEFAULT_PROVIDER, pattern=NAME)  # openai | gemini | deepseek | custom
    model: str | None = Field(None, pattern=NAME)  # defaults to the provider's default model
    max_tokens: int = Field(1024, ge=1, le=4096)
    system_prompt: str | None = Field("You are a helpful assistant.", max_length=4000)


class ChatResponse(BaseModel):
    response: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost_usd: float | None
    finish_reason: str


class RAGRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(3, ge=1, le=20)
    provider: str = Field(DEFAULT_PROVIDER, pattern=NAME)
    model: str | None = Field(None, pattern=NAME)


class CostEstimateRequest(BaseModel):
    model: str = Field(pattern=NAME)
    input_tokens: int = Field(ge=0, le=1_000_000_000)
    output_tokens: int = Field(ge=0, le=1_000_000_000)


def _usage(response) -> tuple[int | None, int | None]:
    u = getattr(response, "usage", None)
    return (getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None)) if u else (None, None)


# ─── Route Handlers ───────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Single-turn chat through the instrumented OpenAI-compatible client (OpenAI, Gemini, DeepSeek, custom)."""
    model = request.model or _model_for(request.provider)
    try:
        client = make_client(request.provider)
        response = client.chat.completions.create(
            model=model,
            max_tokens=request.max_tokens,
            messages=[
                {"role": "system", "content": request.system_prompt or "You are a helpful assistant."},
                {"role": "user", "content": request.message},
            ],
        )
    except ValueError as exc:  # unknown provider or missing key (message names the variable, never its value)
        logger.warning("Chat request rejected: %s", exc)  # operators should see misconfiguration, not just client errors
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Chat request failed: %s", exc, exc_info=True)  # redacted by the logging filter
        raise _provider_error(exc)

    choice = response.choices[0] if getattr(response, "choices", None) else None
    input_tokens, output_tokens = _usage(response)
    cost = CostEstimator.estimate(model, input_tokens, output_tokens) if input_tokens is not None and output_tokens is not None else None
    logger.info("Chat request completed", extra={"model": model, "input_tokens": input_tokens,
                                                 "output_tokens": output_tokens, "cost_usd": cost})
    return ChatResponse(
        response=(choice.message.content or "") if choice else "",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=cost,
        finish_reason=(choice.finish_reason if choice else None) or "unknown",
    )


@router.post("/rag/query")
def rag_query(request: RAGRequest):
    """
    RAG pipeline demo endpoint with multi-stage spans: retrieval, context assembly, generation.
    Retrieval is simulated; replace the marked block with a vector-store query.
    """
    from opentelemetry import trace
    tracer = trace.get_tracer(__name__)
    model = request.model or _model_for(request.provider)

    with tracer.start_as_current_span("rag_pipeline") as pipeline_span:
        pipeline_span.set_attribute("rag.query_chars", len(request.query))  # length + hash, never the text
        pipeline_span.set_attribute("rag.query_sha256", _fingerprint(request.query))
        pipeline_span.set_attribute("rag.top_k", request.top_k)

        with tracer.start_as_current_span("retrieval") as retrieval_span:
            retrieval_span.set_attribute("llm.pipeline.stage", "retrieval")
            # In production: vector DB query here
            docs: list[dict[str, Any]] = [
                {"id": f"doc_{i}", "content": f"Relevant content {i} for: {request.query}", "score": 0.9 - i * 0.1}
                for i in range(request.top_k)
            ]
            retrieval_span.set_attribute("retrieval.documents_returned", len(docs))
            retrieval_span.set_attribute("retrieval.top_score", docs[0]["score"])

        context = "\n\n".join(doc["content"] for doc in docs)

        with tracer.start_as_current_span("generation") as gen_span:
            gen_span.set_attribute("llm.pipeline.stage", "generation")
            try:
                client = make_client(request.provider)
                response = client.chat.completions.create(
                    model=model,
                    max_tokens=512,
                    messages=[
                        {"role": "system", "content": "Answer the question based only on the provided context. Be concise."},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.query}"},
                    ],
                )
            except ValueError as exc:
                logger.warning("RAG request rejected: %s", exc)
                raise HTTPException(status_code=400, detail=str(exc))
            except Exception as exc:
                logger.error("RAG generation failed: %s", exc, exc_info=True)
                raise _provider_error(exc)
            input_tokens, output_tokens = _usage(response)
            if input_tokens is not None:
                gen_span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
                gen_span.set_attribute("gen_ai.usage.output_tokens", output_tokens or 0)

        choice = response.choices[0] if getattr(response, "choices", None) else None
        return {
            "query": request.query,
            "answer": (choice.message.content or "") if choice else "",
            "sources": [{"id": doc["id"], "score": doc["score"]} for doc in docs],
            "model": model,
            "token_usage": {"input": input_tokens, "output": output_tokens},
        }


@router.post("/cost/estimate")
def estimate_cost(request: CostEstimateRequest):
    """Estimate cost for a given model and token counts (approximate price table; see PRICING_VERSION)."""
    cost = CostEstimator.estimate(request.model, request.input_tokens, request.output_tokens)
    return {
        "model": request.model,
        "input_tokens": request.input_tokens,
        "output_tokens": request.output_tokens,
        "estimated_cost_usd": cost,
        "known_model": CostEstimator.is_known(request.model),
    }


@router.get("/cost/monthly-projection")
def monthly_projection(
    model: str = Query("gpt-4o-mini", pattern=NAME),
    daily_requests: int = Query(1000, ge=0, le=1_000_000_000),
    avg_input_tokens: int = Query(500, ge=0, le=1_000_000_000),
    avg_output_tokens: int = Query(200, ge=0, le=1_000_000_000),
):
    """Project monthly cost for capacity planning."""
    monthly = CostEstimator.monthly_budget_usd(daily_requests, avg_input_tokens, avg_output_tokens, model)
    return {
        "model": model,
        "daily_requests": daily_requests,
        "avg_input_tokens": avg_input_tokens,
        "avg_output_tokens": avg_output_tokens,
        "projected_monthly_cost_usd": monthly,
    }
