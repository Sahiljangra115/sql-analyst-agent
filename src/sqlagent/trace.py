"""Optional Langfuse tracing. Off unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set and the
`langfuse` package imports. Any tracing error is swallowed so tracing can never break a query."""
from __future__ import annotations

import contextlib
import os

_client = None
_tried = False


def _lf():
    global _client, _tried
    if not _tried:
        _tried = True
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            try:
                from langfuse import get_client
                _client = get_client()
            except Exception:
                _client = None
    return _client


def enabled() -> bool:
    return _lf() is not None


@contextlib.contextmanager
def span(name: str, input=None, generation: bool = False, model: str | None = None):
    """Yields a `done(output, usage=None)` callable. No-op when Langfuse is off."""
    lf = _lf()
    cm = None
    obj = None
    if lf is not None:
        try:
            cm = (lf.start_as_current_generation(name=name, input=input, model=model) if generation
                  else lf.start_as_current_span(name=name, input=input))
            obj = cm.__enter__()
        except Exception:
            cm = obj = None

    def done(output=None, usage=None):
        if obj is not None:
            try:
                kw = {"output": output}
                if usage:
                    kw["usage_details"] = usage
                obj.update(**kw)
            except Exception:
                pass

    try:
        yield done
    finally:
        if cm is not None:
            try:
                cm.__exit__(None, None, None)
            except Exception:
                pass


def flush():
    lf = _lf()
    if lf is not None:
        try:
            lf.flush()
        except Exception:
            pass
