"""
llm_instrumentor.py
===================
Core OpenTelemetry instrumentation SDK for LLM workloads.

Principal SRE Note:
    This module is the foundation. It sets up the three pillars of observability
    (traces, metrics, logs) with LLM-specific semantics following the
    OpenTelemetry Semantic Conventions for GenAI (semconv 1.26+).

Usage:
    sdk = LLMObservabilitySDK(service_name="my-llm-app")
    sdk.initialize()
"""

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)


@dataclass
class LLMSpanAttributes:
    """
    Semantic conventions for LLM spans.
    Based on OTel GenAI semantic conventions (experimental).
    See: https://opentelemetry.io/docs/specs/semconv/gen-ai/
    """
    # GenAI Span Attributes
    GEN_AI_SYSTEM = "gen_ai.system"
    GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
    GEN_AI_RESPONSE_FINISH_REASON = "gen_ai.response.finish_reasons"
    GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

    # Custom SRE attributes
    LLM_ESTIMATED_COST_USD = "llm.estimated_cost_usd"
    LLM_RETRY_COUNT = "llm.retry_count"
    LLM_PIPELINE_STAGE = "llm.pipeline.stage"
    LLM_RAG_RETRIEVAL_SCORE = "llm.rag.retrieval_score"
    LLM_PROMPT_TOKENS_BUDGET = "llm.prompt_tokens_budget"
    LLM_TOKENS_BUDGET_UTILIZATION = "llm.tokens_budget_utilization"


@dataclass
class SDKConfig:
    """Configuration for the LLM Observability SDK."""
    service_name: str
    service_version: str = "1.0.0"
    environment: str = "development"
    otlp_endpoint: str = "http://localhost:4317"
    export_interval_millis: int = 5000
    enable_cost_tracking: bool = True
    token_budget_alert_threshold: float = 0.8
    extra_resource_attributes: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "SDKConfig":
        """Load config from environment variables (12-factor app pattern)."""
        return cls(
            service_name=os.getenv("OTEL_SERVICE_NAME", "llm-app"),
            service_version=os.getenv("OTEL_SERVICE_VERSION", "1.0.0"),
            environment=os.getenv("ENVIRONMENT", "development"),
            otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
            enable_cost_tracking=os.getenv("LLM_ENABLE_COST_TRACKING", "true").lower() == "true",
            token_budget_alert_threshold=float(os.getenv("LLM_TOKEN_BUDGET_THRESHOLD", "0.8")),
        )


class LLMObservabilitySDK:
    """
    Main entry point for the LLM Observability SDK.

    Initializes all three OTel signals (traces, metrics, logs) and provides
    context managers and decorators for easy instrumentation.

    Example:
        sdk = LLMObservabilitySDK(service_name="rag-pipeline")
        sdk.initialize()

        with sdk.llm_span("generate", model="gpt-4o-mini", system="openai") as span:
            response = client.chat.completions.create(...)
            span.set_attribute("gen_ai.usage.input_tokens", response.usage.input_tokens)
    """

    def __init__(self, service_name: str, **kwargs):
        self.config = SDKConfig(service_name=service_name, **kwargs)
        self._tracer: trace.Tracer | None = None
        self._meter: metrics.Meter | None = None
        self._initialized = False

        # Metric instruments (initialized after SDK setup)
        self._llm_request_duration: Any | None = None
        self._llm_token_counter: Any | None = None
        self._llm_error_counter: Any | None = None
        self._llm_cost_counter: Any | None = None

    def initialize(self) -> None:
        """Initialize all OTel providers and instruments."""
        if self._initialized:
            logger.warning("LLMObservabilitySDK already initialized. Skipping.")
            return

        resource = self._build_resource()
        self._setup_tracing(resource)
        self._setup_metrics(resource)
        self._setup_logging(resource)
        self._create_metric_instruments()

        self._initialized = True
        logger.info(
            "LLMObservabilitySDK initialized",
            extra={"service": self.config.service_name, "endpoint": self.config.otlp_endpoint}
        )

    def _build_resource(self) -> Resource:
        """Build OTel Resource with service metadata."""
        attrs = {
            ResourceAttributes.SERVICE_NAME: self.config.service_name,
            ResourceAttributes.SERVICE_VERSION: self.config.service_version,
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: self.config.environment,
            "team": os.getenv("TEAM_NAME", "platform"),
            "cost_center": os.getenv("COST_CENTER", "engineering"),
        }
        attrs.update(self.config.extra_resource_attributes)
        return Resource.create(attrs)

    def _setup_tracing(self, resource: Resource) -> None:
        """Configure TracerProvider with OTLP exporter."""
        exporter = OTLPSpanExporter(
            endpoint=self.config.otlp_endpoint,
            insecure=True,
        )
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        self._tracer = trace.get_tracer(
            __name__,
            schema_url="https://opentelemetry.io/schemas/1.26.0"
        )

    def _setup_metrics(self, resource: Resource) -> None:
        """Configure MeterProvider with OTLP exporter."""
        exporter = OTLPMetricExporter(
            endpoint=self.config.otlp_endpoint,
            insecure=True,
        )
        reader = PeriodicExportingMetricReader(
            exporter,
            export_interval_millis=self.config.export_interval_millis,
        )
        provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(provider)
        self._meter = metrics.get_meter(__name__)

    def _setup_logging(self, resource: Resource) -> None:
        """Configure LoggerProvider with OTLP exporter and trace correlation."""
        exporter = OTLPLogExporter(
            endpoint=self.config.otlp_endpoint,
            insecure=True,
        )
        provider = LoggerProvider(resource=resource)
        provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
        set_logger_provider(provider)

        # Bridge standard Python logging → OTel logs (with trace_id correlation)
        handler = LoggingHandler(level=logging.INFO, logger_provider=provider)
        logging.getLogger().addHandler(handler)

    def _create_metric_instruments(self) -> None:
        """Create all LLM-specific metric instruments."""
        assert self._meter is not None

        self._llm_request_duration = self._meter.create_histogram(
            name="gen_ai.client.operation.duration",
            description="Duration of LLM API calls",
            unit="s",
        )
        self._llm_token_counter = self._meter.create_counter(
            name="gen_ai.client.token.usage",
            description="Number of tokens used in LLM requests",
            unit="token",
        )
        self._llm_error_counter = self._meter.create_counter(
            name="llm.request.errors",
            description="Number of LLM API errors",
            unit="1",
        )
        self._llm_cost_counter = self._meter.create_counter(
            name="llm.estimated.cost.usd",
            description="Estimated cost of LLM API calls in USD",
            unit="usd",
        )
        self._llm_active_requests = self._meter.create_up_down_counter(
            name="llm.active_requests",
            description="Number of in-flight LLM requests",
            unit="1",
        )

    @contextmanager
    def llm_span(
        self,
        operation_name: str,
        model: str,
        system: str = "unknown",
        pipeline_stage: str | None = None,
        **extra_attrs: Any,
    ) -> Generator[trace.Span, None, None]:
        """
        Context manager that creates a trace span for an LLM operation.

        Args:
            operation_name: e.g., "chat", "embed", "rerank"
            model: Model identifier, e.g., "gpt-4o-mini"
            system: Provider, e.g., "openai", "gemini", "deepseek"
            pipeline_stage: Optional pipeline stage label for RAG pipelines

        Yields:
            An active OTel Span with LLM semantic attributes pre-set.

        Example:
            with sdk.llm_span("chat", model="gpt-4o", system="openai") as span:
                resp = openai_client.chat.completions.create(...)
                span.set_attribute("gen_ai.usage.input_tokens", resp.usage.prompt_tokens)
        """
        assert self._tracer is not None, "SDK not initialized. Call sdk.initialize() first."

        span_name = f"{system} {operation_name}"
        with self._tracer.start_as_current_span(span_name) as span:
            span.set_attribute(LLMSpanAttributes.GEN_AI_SYSTEM, system)
            span.set_attribute(LLMSpanAttributes.GEN_AI_REQUEST_MODEL, model)
            if pipeline_stage:
                span.set_attribute(LLMSpanAttributes.LLM_PIPELINE_STAGE, pipeline_stage)
            for k, v in extra_attrs.items():
                span.set_attribute(k, v)

            metric_labels = {"gen_ai.system": system, "gen_ai.request.model": model}
            if self._llm_active_requests:
                self._llm_active_requests.add(1, metric_labels)

            import time
            start = time.perf_counter()
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as exc:
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)
                if self._llm_error_counter:
                    self._llm_error_counter.add(
                        1,
                        {**metric_labels, "error.type": type(exc).__name__}
                    )
                raise
            finally:
                duration = time.perf_counter() - start
                if self._llm_request_duration:
                    self._llm_request_duration.record(duration, metric_labels)
                if self._llm_active_requests:
                    self._llm_active_requests.add(-1, metric_labels)

    def record_token_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        system: str,
    ) -> None:
        """Record token usage metrics and optionally log cost."""
        labels = {"gen_ai.system": system, "gen_ai.request.model": model}

        if self._llm_token_counter:
            self._llm_token_counter.add(
                input_tokens, {**labels, "gen_ai.token.type": "input"}
            )
            self._llm_token_counter.add(
                output_tokens, {**labels, "gen_ai.token.type": "output"}
            )

        if self.config.enable_cost_tracking and self._llm_cost_counter:
            from src.exporters.cost_estimator import CostEstimator
            cost = CostEstimator.estimate(model, input_tokens, output_tokens)
            self._llm_cost_counter.add(cost, labels)

            # Add cost to current span if active
            current_span = trace.get_current_span()
            if current_span.is_recording():
                current_span.set_attribute(LLMSpanAttributes.LLM_ESTIMATED_COST_USD, cost)

    @property
    def tracer(self) -> trace.Tracer:
        assert self._tracer is not None, "SDK not initialized."
        return self._tracer

    @property
    def meter(self) -> metrics.Meter:
        assert self._meter is not None, "SDK not initialized."
        return self._meter
