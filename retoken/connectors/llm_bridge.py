"""
llm_bridge connector — a PASSIVE metering wrapper for Santander AI Lab's `llm_bridge`
(github.com/SantanderAI/llm_bridge), a vendor-neutral LLM client whose every call returns a
normalised ``LLMResponse(content, model, prompt_tokens, completion_tokens, latency_ms, raw)``.

This is the inline (in-process) attach path, complementing the sidecar log connectors. You
swap one line — wrap your client in :func:`metered` — and every ``chat()`` / ``complete()``
call is logged to TokenLedger and reconciled (output re-tokenised from the returned text,
input bounded), with NO change to the rest of your application and NO effect on the call.

Why this exists (see docs/research-tokenledger-landscape.md): gateways like LiteLLM/Helicone
expose callback hooks but hand back the PROVIDER's own reported usage — they aggregate and
trust. TokenLedger attaches at the same response layer and adds INDEPENDENT re-tokenisation on
top. That verify-vs-aggregate gap is the wedge; this connector is how it reaches llm_bridge.

Architectural non-negotiables honoured here:
  - Passive: the real ``inner.chat`` runs first and its errors propagate untouched; recording
    and reconciliation are wrapped so a logging failure can NEVER break the caller's call.
  - No egress: usage enrichment from ``resp.raw`` is pure local parsing (recorder adapters);
    nothing here makes a network call or hits a provider count-tokens API.
  - Honest confidence: counting/labels are core's job; this only normalises into a CallRecord.

``llm_bridge`` is an OPTIONAL dependency. When installed, the wrapper subclasses its
``LLMClient`` so it is a true drop-in (isinstance-compatible; the base ``complete()`` dispatches
to our metered ``chat``). When absent, it degrades to a duck-typed wrapper so TokenLedger never
hard-depends on llm_bridge — it works against anything exposing the same tiny interface.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional

from ..core import CallRecord, CallReconciliation, Usage, reconcile_call
from ..quality import QualitySignal
from ..recorder import from_anthropic, from_openai
from ..store import Store

# Optional drop-in base. Subclass the real LLMClient when available; else fall back to object.
try:  # pragma: no cover - presence depends on the host environment
    from llm_bridge import LLMClient as _LLMClientBase  # type: ignore
    _HAVE_BRIDGE = True
except Exception:  # pragma: no cover
    _LLMClientBase = object  # type: ignore
    _HAVE_BRIDGE = False


def _messages_to_text(messages: Any) -> str:
    """Flatten a chat messages list into plain text for re-tokenisation (input bounding)."""
    parts: List[str] = []
    for m in messages or []:
        if isinstance(m, dict):
            c = m.get("content")
            parts.append(c if isinstance(c, str) else json.dumps(c))
        else:
            parts.append(str(m))
    return "\n".join(parts)


def _usage_from_response(resp: Any, provider: str) -> Usage:
    """Build a Usage from a normalised LLMResponse.

    The normalised response is the source of truth for input/output tokens. Reasoning and cache
    tokens are NOT on the normalised contract — they live only in the provider's raw payload
    (``resp.raw``) — so we enrich those buckets from there with the same local adapters the
    recorder uses. Pure parsing, no network, best-effort: enrichment never raises.
    """
    base = Usage(
        input_tokens=int(getattr(resp, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(resp, "completion_tokens", 0) or 0),
    )
    raw = getattr(resp, "raw", None)
    if raw is None:
        return base
    try:
        raw_usage = getattr(raw, "usage", None)
        if raw_usage is None and isinstance(raw, dict):
            raw_usage = raw.get("usage")
        if raw_usage is None:
            return base
        p = (provider or "").lower()
        enr = from_anthropic(raw_usage) if ("anthropic" in p or "claude" in p) else from_openai(raw_usage)
        # Only fill the buckets the normalised response omits; never override its input/output.
        base.reasoning_tokens = enr.reasoning_tokens or base.reasoning_tokens
        base.cache_read_tokens = enr.cache_read_tokens or base.cache_read_tokens
        base.cache_creation_tokens = enr.cache_creation_tokens or base.cache_creation_tokens
    except Exception:
        pass
    return base


class MeteringLLMClient(_LLMClientBase):  # type: ignore[misc,valid-type]
    """Passive metering wrapper around any llm_bridge ``LLMClient`` (or duck-typed equivalent).

    Every ``chat`` / ``complete`` is logged to ``store`` and reconciled. Per-call attribution can
    be supplied with ``tl_user_id`` / ``tl_session_id`` / ``tl_task_class`` / ``tl_tags`` /
    ``tl_quality`` kwargs (named params, so the inner client never sees them). Pass ``on_call`` to
    receive each :class:`CallReconciliation` live — e.g. to alert on an output over-count.
    """

    def __init__(
        self,
        inner: Any,
        store: Store,
        *,
        user_id: str = "default",
        session_id: str = "default",
        route: Optional[str] = None,
        on_call: Optional[Callable[[CallReconciliation], None]] = None,
        reconcile: bool = True,
        now: Optional[Callable[[], str]] = None,
    ) -> None:
        self._inner = inner
        self._store = store
        self._user_id = user_id
        self._session_id = session_id
        self._route = route
        self._on_call = on_call
        self._reconcile = reconcile
        self._now = now or (lambda: datetime.now(timezone.utc).isoformat())

    # --- identity: behave like the wrapped client -------------------------------------
    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def provider(self) -> str:
        return getattr(self._inner, "provider", "unknown")

    def __getattr__(self, name: str) -> Any:
        # Delegate anything we don't define to the inner client (but chat/complete are
        # overridden above, so they always hit the metered path). Guard against recursion
        # before _inner is bound.
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)

    # --- the metered path -------------------------------------------------------------
    def chat(
        self,
        messages: List[dict],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tl_user_id: Optional[str] = None,
        tl_session_id: Optional[str] = None,
        tl_task_class: Optional[str] = None,
        tl_tags: Any = None,
        tl_quality: Optional[QualitySignal] = None,
        **kwargs: Any,
    ) -> Any:
        # 1) The REAL call first. If it raises, that is the call failing — let it propagate.
        #    tl_* are named params (structurally absent from **kwargs), so the inner never sees them.
        resp = self._inner.chat(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)
        # 2) Passive logging + reconciliation. Must NEVER break the caller.
        try:
            self._record(messages, resp, tl_user_id, tl_session_id, tl_task_class, tl_tags, tl_quality)
        except Exception:
            pass
        return resp

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Any:
        # Route single-turn calls through our OWN chat so they are metered too. Calling
        # inner.complete would bypass metering (the base complete dispatches to inner.chat).
        messages: List[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)

    # --- internal --------------------------------------------------------------------
    def _record(
        self,
        messages: List[dict],
        resp: Any,
        tl_user_id: Optional[str],
        tl_session_id: Optional[str],
        tl_task_class: Optional[str],
        tl_tags: Any,
        tl_quality: Optional[QualitySignal] = None,
    ) -> Optional[CallReconciliation]:
        provider = self.provider
        model = getattr(resp, "model", None) or self._inner.model
        rec = CallRecord(
            provider=provider,
            model=model,
            route=self._route or "/chat/completions",
            user_id=str(tl_user_id or self._user_id),
            session_id=str(tl_session_id or self._session_id),
            ts=self._now(),
            reported=_usage_from_response(resp, provider),
            request_text=_messages_to_text(messages),
            response_text=getattr(resp, "content", "") or "",
            latency_ms=getattr(resp, "latency_ms", None),
            task_class=tl_task_class or "unclassified",
            quality=tl_quality,
        )
        if rec.task_class == "unclassified":
            from ..classify import classify_call
            rec.task_class = classify_call(rec, tl_tags)
        self._store.record(rec)
        if self._reconcile or self._on_call:
            # Live reconcile uses the TRUE message count for input-overhead. (A later dashboard
            # reconcile defaults to num_messages=1; the input band's tolerance absorbs the
            # ~3-tokens/message difference, so the two never disagree on a verdict.)
            rc = reconcile_call(rec, num_messages=max(1, len(messages or [])))
            if self._on_call:
                try:
                    self._on_call(rc)
                except Exception:
                    pass
            return rc
        return None


def metered(
    inner: Any,
    store: Store,
    *,
    user_id: str = "default",
    session_id: str = "default",
    route: Optional[str] = None,
    on_call: Optional[Callable[[CallReconciliation], None]] = None,
    reconcile: bool = True,
) -> MeteringLLMClient:
    """Wrap an llm_bridge ``LLMClient`` so every call is metered. Returns a drop-in client.

    Example::

        from llm_bridge import create_llm
        from tokenledger.store import Store
        from tokenledger.connectors import metered

        llm = metered(create_llm({"provider": "openai", "model": "gpt-4o-mini"}),
                      Store("tokenledger.db"), session_id="prod")
        llm.complete("Hello!")   # logged + reconciled, your code is otherwise unchanged
    """
    return MeteringLLMClient(
        inner, store, user_id=user_id, session_id=session_id,
        route=route, on_call=on_call, reconcile=reconcile,
    )
