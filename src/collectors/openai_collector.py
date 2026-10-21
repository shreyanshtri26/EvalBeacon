"""
openai_collector.py
===================
Drop-in replacement for the OpenAI client with full OTel instrumentation.
Also serves DeepSeek and Gemini, which expose OpenAI-compatible endpoints
(see src/collectors/providers.py); the span name and gen_ai.system carry the provider.

Usage:
    from src.collectors.openai_collector import InstrumentedOpenAI
    client = InstrumentedOpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}]
    )
"""

import logging
import time
from typing import Any

import openai
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from src.collectors.llm_instrumentor import LLMSpanAttributes

logger = logging.getLogger(__name__)

_sdk = None


def setup_sdk(sdk) -> None:
    global _sdk
    _sdk = sdk


class InstrumentedChatCompletions:
    """Wraps openai.resources.chat.Completions with OTel instrumentation."""

    def __init__(self, original_completions, sdk=None, provider: str = "openai"):
        self._completions = original_completions
        self._sdk = sdk or _sdk
        self._provider = provider

    def create(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        stream: bool = False,
        **kwargs,
    ) -> Any:
        tracer = trace.get_tracer(__name__)

        with tracer.start_as_current_span(f"{self._provider} chat") as span:
            span.set_attribute(LLMSpanAttributes.GEN_AI_SYSTEM, self._provider)
            span.set_attribute(LLMSpanAttributes.GEN_AI_REQUEST_MODEL, model)
            span.set_attribute("gen_ai.request.message_count", len(messages))
            if max_tokens:
                span.set_attribute(LLMSpanAttributes.GEN_AI_REQUEST_MAX_TOKENS, max_tokens)
            if temperature is not None:
                span.set_attribute(LLMSpanAttributes.GEN_AI_REQUEST_TEMPERATURE, temperature)

            start = time.perf_counter()
            try:
                call_kwargs: dict = {"model": model, "messages": messages, **kwargs}
                if max_tokens:
                    call_kwargs["max_tokens"] = max_tokens
                if temperature is not None:
                    call_kwargs["temperature"] = temperature
                if stream:
                    call_kwargs["stream"] = stream

                response = self._completions.create(**call_kwargs)
                duration = time.perf_counter() - start

                if not stream:
                    try:  # telemetry must never corrupt a successful application result
                        self._record_response_attributes(span, response, model, duration)
                    except Exception:  # noqa: BLE001 - e.g. missing/malformed usage
                        span.set_attribute("llm.telemetry_incomplete", True)
                        logger.warning("Could not record response attributes", exc_info=True)

                span.set_status(Status(StatusCode.OK))
                return response

            except openai.RateLimitError as exc:
                span.set_status(Status(StatusCode.ERROR, "rate_limit"))
                span.record_exception(exc)
                span.set_attribute("error.type", "rate_limit")
                raise
            except openai.APITimeoutError as exc:
                span.set_status(Status(StatusCode.ERROR, "timeout"))
                span.record_exception(exc)
                span.set_attribute("error.type", "timeout")
                raise
            except Exception as exc:
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)
                span.set_attribute("error.type", type(exc).__name__)
                raise

    def _record_response_attributes(self, span, response, model: str, duration: float) -> None:
        choice = response.choices[0] if response.choices else None
        if choice:
            span.set_attribute(
                LLMSpanAttributes.GEN_AI_RESPONSE_FINISH_REASON,
                choice.finish_reason or "unknown"
            )
        span.set_attribute(LLMSpanAttributes.GEN_AI_USAGE_INPUT_TOKENS, response.usage.prompt_tokens)
        span.set_attribute(LLMSpanAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, response.usage.completion_tokens)
        span.set_attribute("llm.response.duration_seconds", round(duration, 4))

        if self._sdk:
            self._sdk.record_token_usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                model=model,
                system=self._provider,
            )

        logger.info(
            "LLM API call completed",
            extra={
                "model": model,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "duration_seconds": round(duration, 4),
            }
        )


class InstrumentedOpenAI:
    """
    Instrumented drop-in replacement for openai.OpenAI().

    Example:
        client = InstrumentedOpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Explain SLOs"}]
        )
    """

    def __init__(self, sdk=None, provider: str = "openai", **openai_kwargs):
        self._client = openai.OpenAI(**openai_kwargs)
        self.chat = _ChatProxy(self._client.chat, sdk=sdk, provider=provider)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _ChatProxy:
    def __init__(self, original_chat, sdk=None, provider: str = "openai"):
        self._chat = original_chat
        self.completions = InstrumentedChatCompletions(original_chat.completions, sdk=sdk, provider=provider)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)
