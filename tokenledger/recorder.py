"""
Recorder: turn a real provider API response into a CallRecord and persist it.

This is the piece that "runs across every session / API call". Two ways to use it:

1. Adapter functions (from_openai / from_anthropic) that parse a raw response object or
   dict into a normalized Usage. Call these from your own SDK wrapper or gateway hook.
2. wrap_openai() / wrap_anthropic(): monkeypatch-style passive interceptors you can attach
   once at process start so every call in a session is logged with no per-call changes.

Nothing here changes the request or response. Logging is passive and best-effort: a
logging failure must never break the actual call.
"""

from __future__ import annotations

from typing import Any, Optional

from .core import CallRecord, Usage
from .quality import QualitySignal
from .store import Store


def _g(obj: Any, *names: str, default: Any = 0) -> Any:
    """Get an attribute or dict key, trying several names."""
    for n in names:
        if obj is None:
            return default
        if isinstance(obj, dict):
            if n in obj and obj[n] is not None:
                return obj[n]
        else:
            v = getattr(obj, n, None)
            if v is not None:
                return v
    return default


def from_openai(usage: Any) -> Usage:
    """Parse an OpenAI-style usage object/dict (chat.completions, responses API) into the canonical
    DISJOINT Usage (see core.Usage). OpenAI reports OVERLAPPING buckets: `prompt_tokens` INCLUDES
    cached, `completion_tokens` INCLUDES reasoning. We subtract the subsets so each token is counted
    ONCE (otherwise cost double-counts cache and reasoning)."""
    details = _g(usage, "completion_tokens_details", default={}) or {}
    prompt_details = _g(usage, "prompt_tokens_details", default={}) or {}
    prompt = int(_g(usage, "prompt_tokens", "input_tokens"))
    completion = int(_g(usage, "completion_tokens", "output_tokens"))
    reasoning = int(_g(details, "reasoning_tokens"))
    cached = int(_g(prompt_details, "cached_tokens"))
    return Usage(
        input_tokens=max(0, prompt - cached),         # uncached input only (cached counted below)
        output_tokens=max(0, completion - reasoning),  # visible output only (reasoning counted below)
        reasoning_tokens=reasoning,
        cache_read_tokens=cached,
    )


def from_anthropic(usage: Any) -> Usage:
    """Parse an Anthropic-style usage object/dict. Thinking tokens are billed as output."""
    return Usage(
        input_tokens=int(_g(usage, "input_tokens")),
        output_tokens=int(_g(usage, "output_tokens")),
        cache_read_tokens=int(_g(usage, "cache_read_input_tokens")),
        cache_creation_tokens=int(_g(usage, "cache_creation_input_tokens")),
    )


_ADAPTERS = {"openai": from_openai, "anthropic": from_anthropic}


def record_call(
    store: Store,
    *,
    provider: str,
    model: str,
    user_id: str,
    session_id: str,
    ts: str,
    usage: Any,
    request_text: str = "",
    response_text: str = "",
    route: str = "/chat/completions",
    latency_ms: Optional[float] = None,
    task_class: Optional[str] = None,
    tags: Any = None,
    quality: Optional[QualitySignal] = None,
    reported_cost_usd: Optional[float] = None,
) -> CallRecord:
    """Normalize a provider usage payload into a CallRecord and persist it.

    task_class: explicit activity type if known; else pass tags/metadata (or nothing) and it is
    classified LOCALLY (see classify.py).
    quality: optional per-call quality signal (cost-per-accepted). Usually unknown at call time —
    attach it later with store.set_quality(call_id, signal) / store.set_quality_by(...).
    """
    # If already normalized, use it as-is. Only run a provider adapter on a raw payload.
    if isinstance(usage, Usage):
        norm = usage
    else:
        adapter = _ADAPTERS.get(provider)
        norm = adapter(usage) if adapter else _coerce_usage(usage)
    rec = CallRecord(
        provider=provider, model=model, route=route, user_id=user_id,
        session_id=session_id, ts=ts, reported=norm,
        request_text=request_text, response_text=response_text, latency_ms=latency_ms,
        task_class=task_class or "unclassified",
        quality=quality,
        reported_cost_usd=reported_cost_usd,
    )
    if rec.task_class == "unclassified":
        from .classify import classify_call
        rec.task_class = classify_call(rec, tags)
    try:
        store.record(rec)
    except Exception:
        pass  # passive logging must never break the caller
    return rec


def _coerce_usage(usage: Any) -> Usage:
    if isinstance(usage, Usage):
        return usage
    return from_openai(usage)  # OpenAI shape is the common default
