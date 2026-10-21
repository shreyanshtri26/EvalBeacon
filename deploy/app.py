"""Slim Vercel entry point: dashboard + eval API only (no OTLP exporters or LLM clients in the bundle)."""
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from src.evalbeacon.api import install_error_handlers, router
from src.evalbeacon.redact import install_log_redaction

install_log_redaction()
trace.set_tracer_provider(TracerProvider())  # real trace ids on every case, no exporter

app = FastAPI(title="EvalBeacon", version="1.0.0")
app.include_router(router)
install_error_handlers(app)


@app.get("/health")
def health():
    return {"status": "healthy", "service": "evalbeacon"}
