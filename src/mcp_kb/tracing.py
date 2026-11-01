"""Phoenix tracing — optional. If Phoenix isn't running, calls are no-ops."""
import os
from contextlib import contextmanager

_tracer = None


def setup_tracing() -> bool:
    """Wire up Phoenix if PHOENIX_ENABLED=1 (or just present). Returns True if active."""
    if not os.environ.get("PHOENIX_ENABLED"):
        return False
    try:
        import phoenix as px
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from phoenix.otel import register

        register(project_name="mcp-knowledge-base", auto_instrument=False)
        global _tracer
        _tracer = trace.get_tracer("mcp-knowledge-base")
        return True
    except Exception:
        return False


@contextmanager
def search_span(query: str):
    """Context manager that records a retrieval span when tracing is active."""
    if _tracer is None:
        yield _SpanStub()
        return
    from opentelemetry.trace import SpanKind
    with _tracer.start_as_current_span("retrieval", kind=SpanKind.INTERNAL) as span:
        span.set_attribute("query", query)
        wrapper = _SpanWrapper(span)
        yield wrapper


class _SpanStub:
    def record_results(self, results): pass


class _SpanWrapper:
    def __init__(self, span): self._span = span

    def record_results(self, results: list[dict]):
        self._span.set_attribute("retrieved_count", len(results))
        for i, r in enumerate(results):
            self._span.set_attribute(f"result.{i}.source", r["source"])
            self._span.set_attribute(f"result.{i}.citation", r["citation"])
