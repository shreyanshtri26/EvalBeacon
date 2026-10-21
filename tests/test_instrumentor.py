"""
test_instrumentor.py
====================
Unit tests for the LLM Observability SDK core components.

Run with:
    pytest tests/ -v --cov=src --cov-report=term-missing
"""

from unittest.mock import MagicMock, patch

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.collectors.llm_instrumentor import LLMObservabilitySDK, LLMSpanAttributes, SDKConfig
from src.exporters.cost_estimator import CostEstimator

# ─── Fixtures ─────────────────────────────────────────────────────────────────

# OTel allows setting the global TracerProvider only once per process.
_EXPORTER = InMemorySpanExporter()
_PROVIDER = TracerProvider()
_PROVIDER.add_span_processor(SimpleSpanProcessor(_EXPORTER))
trace.set_tracer_provider(_PROVIDER)


@pytest.fixture
def span_exporter():
    """In-memory span exporter for test assertions (cleared per test)."""
    _EXPORTER.clear()
    return _EXPORTER


@pytest.fixture
def tracer_provider(span_exporter):
    return _PROVIDER


@pytest.fixture
def mock_sdk():
    """Minimal SDK mock for collector tests."""
    sdk = MagicMock()
    sdk.record_token_usage = MagicMock()
    return sdk


# ─── SDKConfig Tests ──────────────────────────────────────────────────────────

class TestSDKConfig:
    def test_default_values(self):
        config = SDKConfig(service_name="test-service")
        assert config.service_name == "test-service"
        assert config.service_version == "1.0.0"
        assert config.environment == "development"
        assert config.otlp_endpoint == "http://localhost:4317"
        assert config.enable_cost_tracking is True
        assert config.token_budget_alert_threshold == 0.8

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("OTEL_SERVICE_NAME", "env-service")
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317")
        monkeypatch.setenv("LLM_TOKEN_BUDGET_THRESHOLD", "0.9")

        config = SDKConfig.from_env()
        assert config.service_name == "env-service"
        assert config.environment == "production"
        assert config.otlp_endpoint == "http://collector:4317"
        assert config.token_budget_alert_threshold == 0.9

    def test_from_env_defaults(self, monkeypatch):
        """Should use defaults when env vars are not set."""
        monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
        config = SDKConfig.from_env()
        assert config.service_name == "llm-app"


# ─── CostEstimator Tests ──────────────────────────────────────────────────────

class TestCostEstimator:
    def test_openai_mini_cost(self):
        # 1M input + 0 output = $0.15
        cost = CostEstimator.estimate("gpt-4o-mini", 1_000_000, 0)
        assert cost == pytest.approx(0.15, rel=1e-3)

    def test_deepseek_and_gemini_are_priced(self):
        assert CostEstimator.estimate("deepseek-chat", 1_000_000, 1_000_000) == pytest.approx(1.37, rel=1e-3)
        assert CostEstimator.estimate("gemini-2.0-flash", 1_000_000, 1_000_000) == pytest.approx(0.50, rel=1e-3)

    def test_openai_gpt4o_cost(self):
        cost = CostEstimator.estimate("gpt-4o", 100_000, 50_000)
        assert cost == pytest.approx(0.75, rel=1e-3)  # 0.25 + 0.50

    def test_zero_tokens(self):
        cost = CostEstimator.estimate("gpt-4o-mini", 0, 0)
        assert cost == 0.0

    def test_unknown_model_uses_fallback(self):
        cost_unknown = CostEstimator.estimate("unknown-model-xyz", 1000, 500)
        cost_fallback = CostEstimator.estimate("unknown", 1000, 500)
        assert cost_unknown == cost_fallback

    def test_prefix_matching(self):
        """Model variants should match base model pricing."""
        base = CostEstimator.estimate("gpt-4o-mini", 1000, 1000)
        assert CostEstimator.estimate("gpt-4o-mini-2024-07-18", 1000, 1000) == base
        # longest prefix wins: a mini variant must not be priced as full gpt-4o
        assert base != CostEstimator.estimate("gpt-4o", 1000, 1000)

    def test_monthly_projection(self):
        monthly = CostEstimator.monthly_budget_usd(
            daily_requests=1000,
            avg_input_tokens=500,
            avg_output_tokens=200,
            model="gpt-4o-mini",
        )
        # 1000 req/day * 30 days * cost_per_request
        expected_per_req = CostEstimator.estimate("gpt-4o-mini", 500, 200)
        assert monthly == pytest.approx(expected_per_req * 1000 * 30, rel=1e-3)

    def test_get_all_models_excludes_unknown(self):
        models = CostEstimator.get_all_models()
        assert "unknown" not in models
        assert "gemini-2.0-flash" in models and "deepseek-chat" in models
        assert "gpt-4o" in models


# ─── LLM Span Context Manager Tests ──────────────────────────────────────────

class TestLLMSpanContextManager:
    @pytest.fixture(autouse=True)
    def setup_sdk(self, tracer_provider):
        """Set up a real SDK using the test tracer provider."""
        self.sdk = LLMObservabilitySDK.__new__(LLMObservabilitySDK)
        self.sdk.config = SDKConfig(service_name="test")
        self.sdk._tracer = tracer_provider.get_tracer("test")
        self.sdk._initialized = True
        self.sdk._llm_request_duration = MagicMock()
        self.sdk._llm_error_counter = MagicMock()
        self.sdk._llm_active_requests = MagicMock()
        self.sdk._llm_token_counter = MagicMock()
        self.sdk._llm_cost_counter = MagicMock()

    def test_span_created_with_correct_attributes(self, span_exporter):
        with self.sdk.llm_span("chat", model="gpt-4o-mini", system="openai"):
            pass

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "openai chat"
        assert span.attributes[LLMSpanAttributes.GEN_AI_SYSTEM] == "openai"
        assert span.attributes[LLMSpanAttributes.GEN_AI_REQUEST_MODEL] == "gpt-4o-mini"

    def test_span_records_error_on_exception(self, span_exporter):
        with pytest.raises(ValueError):
            with self.sdk.llm_span("chat", model="gpt-4o", system="openai"):
                raise ValueError("LLM call failed")

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        from opentelemetry.trace import StatusCode
        assert span.status.status_code == StatusCode.ERROR

    def test_pipeline_stage_attribute(self, span_exporter):
        with self.sdk.llm_span("embed", model="text-embedding-3-small", system="openai", pipeline_stage="retrieval"):
            pass

        spans = span_exporter.get_finished_spans()
        assert spans[0].attributes[LLMSpanAttributes.LLM_PIPELINE_STAGE] == "retrieval"

    def test_active_requests_incremented_and_decremented(self):
        with self.sdk.llm_span("chat", model="test-model", system="test"):
            self.sdk._llm_active_requests.add.assert_called_once()

        # Should be called twice: once to add, once to subtract
        assert self.sdk._llm_active_requests.add.call_count == 2


# ─── Provider Collector Tests ─────────────────────────────────────────────────

class TestOpenAICompatibleCollector:
    @pytest.mark.parametrize("provider", ["openai", "gemini", "deepseek", "anthropic"])
    def test_span_carries_provider_and_usage(self, span_exporter, provider):
        resp = MagicMock()
        resp.usage.prompt_tokens, resp.usage.completion_tokens = 100, 50
        resp.choices = [MagicMock(finish_reason="stop")]

        with patch("openai.OpenAI") as cls:
            cls.return_value.chat.completions.create.return_value = resp
            from src.collectors.openai_collector import InstrumentedOpenAI
            client = InstrumentedOpenAI(provider=provider, api_key="test")
            out = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])

        assert out is resp
        (span,) = [s for s in span_exporter.get_finished_spans() if s.name == f"{provider} chat"]
        assert span.attributes[LLMSpanAttributes.GEN_AI_SYSTEM] == provider
        assert span.attributes[LLMSpanAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 100
        assert span.attributes[LLMSpanAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 50

    def test_make_client_validates_provider_and_key(self, monkeypatch):
        from src.collectors import providers
        with pytest.raises(ValueError, match="unknown provider"):
            providers.make_client("not-a-provider")
        monkeypatch.setattr(providers, "load_dotenv", lambda: None)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("DeepSeek_API_KEY", raising=False)
        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY is not set"):
            providers.make_client("deepseek")

    def test_key_lookup_is_case_insensitive(self, monkeypatch):
        from src.collectors import providers
        monkeypatch.setattr(providers, "load_dotenv", lambda: None)
        monkeypatch.setenv("Gemini_API_KEY", "k")
        assert providers.api_key("gemini") == "k"


# ─── Integration-style Tests ──────────────────────────────────────────────────

class TestEndToEnd:
    def test_cost_flows_into_span(self, span_exporter, tracer_provider):
        """Token usage recorded via SDK should attach cost to the active span."""
        sdk = LLMObservabilitySDK.__new__(LLMObservabilitySDK)
        sdk.config = SDKConfig(service_name="test", enable_cost_tracking=True)
        sdk._tracer = tracer_provider.get_tracer("test")
        sdk._initialized = True
        sdk._llm_request_duration = MagicMock()
        sdk._llm_error_counter = MagicMock()
        sdk._llm_active_requests = MagicMock()
        sdk._llm_token_counter = MagicMock()
        sdk._llm_cost_counter = MagicMock()

        with sdk.llm_span("chat", model="gpt-4o-mini", system="openai"):
            sdk.record_token_usage(1000, 500, "gpt-4o-mini", "openai")

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert LLMSpanAttributes.LLM_ESTIMATED_COST_USD in spans[0].attributes
