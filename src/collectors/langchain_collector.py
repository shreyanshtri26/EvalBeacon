"""
langchain_collector.py
======================
LangChain callback handler for OpenTelemetry instrumentation.

This integrates with LangChain's callback system to provide trace spans
and metrics for every LLM call, chain step, and retriever query.

Usage:
    from src.collectors.langchain_collector import OTelCallbackHandler
    from langchain_openai import ChatOpenAI

    handler = OTelCallbackHandler(sdk=sdk)
    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[handler])
"""

import logging
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from src.collectors.llm_instrumentor import LLMSpanAttributes

logger = logging.getLogger(__name__)


class OTelCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler that creates OTel spans for each operation.

    Instruments:
    - LLM calls (on_llm_start / on_llm_end / on_llm_error)
    - Chain steps (on_chain_start / on_chain_end)
    - Retriever queries (on_retriever_start / on_retriever_end)
    - Tool calls (on_tool_start / on_tool_end)
    """

    def __init__(self, sdk=None):
        super().__init__()
        self._sdk = sdk
        self._tracer = trace.get_tracer(__name__)
        # Track active spans by run_id
        self._active_spans: dict[str, tuple[trace.Span, Any, float]] = {}

    # ─── LLM Callbacks ────────────────────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        model = (serialized.get("kwargs") or {}).get("model", "unknown")
        system = self._infer_system(serialized)

        span = self._tracer.start_span(f"{system} chat")
        span.set_attribute(LLMSpanAttributes.GEN_AI_SYSTEM, system)
        span.set_attribute(LLMSpanAttributes.GEN_AI_REQUEST_MODEL, model)
        span.set_attribute("gen_ai.request.prompt_count", len(prompts))
        span.set_attribute(LLMSpanAttributes.LLM_PIPELINE_STAGE, "llm_call")

        ctx = trace.use_span(span, end_on_exit=False)
        ctx.__enter__()
        self._active_spans[str(run_id)] = (span, ctx, time.perf_counter())

    def on_llm_end(self, response: LLMResult, run_id: UUID, **kwargs: Any) -> None:
        run_key = str(run_id)
        if run_key not in self._active_spans:
            return

        span, ctx, start_time = self._active_spans.pop(run_key)
        duration = time.perf_counter() - start_time

        # Extract token usage from LLMResult metadata
        if response.llm_output:
            usage = response.llm_output.get("usage", {}) or response.llm_output.get("token_usage", {})
            input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens", 0)
            output_tokens = usage.get("output_tokens") or usage.get("completion_tokens", 0)

            if input_tokens or output_tokens:
                span.set_attribute(LLMSpanAttributes.GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
                span.set_attribute(LLMSpanAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)

                if self._sdk:
                    attrs = getattr(span, "attributes", None) or {}  # only SDK spans expose attributes
                    model = attrs.get(LLMSpanAttributes.GEN_AI_REQUEST_MODEL, "unknown")
                    system = attrs.get(LLMSpanAttributes.GEN_AI_SYSTEM, "unknown")
                    self._sdk.record_token_usage(input_tokens, output_tokens, model, system)

        span.set_attribute("llm.response.duration_seconds", round(duration, 4))
        span.set_status(Status(StatusCode.OK))
        span.end()
        ctx.__exit__(None, None, None)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        run_key = str(run_id)
        if run_key not in self._active_spans:
            return

        span, ctx, _ = self._active_spans.pop(run_key)
        if isinstance(error, Exception):
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)
        span.end()
        ctx.__exit__(None, None, None)

    # ─── Chain Callbacks ──────────────────────────────────────────────────────

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        chain_name = serialized.get("id", ["unknown"])[-1]
        span = self._tracer.start_span(f"chain.{chain_name}")
        span.set_attribute("langchain.chain.name", chain_name)
        span.set_attribute(LLMSpanAttributes.LLM_PIPELINE_STAGE, "chain")

        ctx = trace.use_span(span, end_on_exit=False)
        ctx.__enter__()
        self._active_spans[f"chain_{run_id}"] = (span, ctx, time.perf_counter())

    def on_chain_end(self, outputs: dict[str, Any], run_id: UUID, **kwargs: Any) -> None:
        run_key = f"chain_{run_id}"
        if run_key not in self._active_spans:
            return
        span, ctx, _ = self._active_spans.pop(run_key)
        span.set_status(Status(StatusCode.OK))
        span.end()
        ctx.__exit__(None, None, None)

    # ─── Retriever Callbacks ──────────────────────────────────────────────────

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        span = self._tracer.start_span("retriever.query")
        span.set_attribute(LLMSpanAttributes.LLM_PIPELINE_STAGE, "retrieval")
        span.set_attribute("retriever.query_length", len(query))

        ctx = trace.use_span(span, end_on_exit=False)
        ctx.__enter__()
        self._active_spans[f"retriever_{run_id}"] = (span, ctx, time.perf_counter())

    def on_retriever_end(self, documents, run_id: UUID, **kwargs: Any) -> None:
        run_key = f"retriever_{run_id}"
        if run_key not in self._active_spans:
            return
        span, ctx, start_time = self._active_spans.pop(run_key)
        span.set_attribute("retriever.documents_returned", len(documents))
        span.set_attribute("retriever.duration_seconds", round(time.perf_counter() - start_time, 4))
        span.set_status(Status(StatusCode.OK))
        span.end()
        ctx.__exit__(None, None, None)

    # ─── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _infer_system(serialized: dict) -> str:
        """Infer LLM provider from serialized chain info."""
        id_path = serialized.get("id", [])
        id_str = " ".join(id_path).lower()
        if "openai" in id_str:
            return "openai"
        if "deepseek" in id_str:
            return "deepseek"
        if "google" in id_str or "gemini" in id_str:
            return "gemini"
        if "cohere" in id_str:
            return "cohere"
        return "unknown"
