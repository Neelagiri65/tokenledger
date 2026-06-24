"""
LiteLLM connector. Reads a LiteLLM SpendLogs export (JSONL — one JSON object per line) and
normalises each row into a TokenLedger CallRecord. This is the sidecar path: LiteLLM already
writes spend logs; TokenLedger reads them and audits the numbers from the outside.

LiteLLM spend-log fields vary by version/config. We read defensively. If prompts/responses are
present (store_prompts_in_spend_logs enabled), output tokens get an EXACT re-count; if not, the
record still reconciles on reported counts (no exact output check, flagged accordingly).
"""

from __future__ import annotations

import json
from typing import Any, Iterator, Optional

from ..core import CallRecord, Usage
from ..store import Store


def _first(d: dict, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _to_float(value: Any) -> Optional[float]:
    """Coerce a reported-cost field to float, or None if absent/unparseable/non-finite (never raise).
    Rejects inf/nan so a bad value can't poison a summed money figure."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    import math
    return f if math.isfinite(f) else None


def _text_from(value: Any) -> str:
    """Coerce a messages list / response object / string into plain text for re-tokenisation."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):  # chat messages
        parts = []
        for m in value:
            if isinstance(m, dict):
                c = m.get("content")
                parts.append(c if isinstance(c, str) else json.dumps(c))
            else:
                parts.append(str(m))
        return "\n".join(parts)
    if isinstance(value, dict):  # response object
        choices = value.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
        return json.dumps(value)
    return str(value)


def parse_litellm_row(row: dict) -> Optional[CallRecord]:
    """Map one LiteLLM spend-log dict to a CallRecord. Returns None if it isn't a usable row."""
    model = _first(row, "model", "model_name")
    if not model:
        return None

    metadata = row.get("metadata") or {}
    prompt = int(_first(row, "prompt_tokens", "input_tokens", default=0) or 0)
    completion = int(_first(row, "completion_tokens", "output_tokens", default=0) or 0)
    reasoning = int(_first(row, "reasoning_tokens", default=0) or 0)
    cache_read = int(_first(row, "cache_read_input_tokens", "cached_tokens", default=0) or 0)
    if not cache_read and _first(row, "cache_hit") in (True, "true", "True"):
        # Only a boolean cache hit -> treat the whole prompt as cached (best available split).
        cache_read = prompt
    # Normalise to the canonical DISJOINT Usage: prompt INCLUDES cached, completion INCLUDES
    # reasoning (OpenAI-shaped) -> subtract the subsets so cost counts each token once.
    usage = Usage(
        input_tokens=max(0, prompt - cache_read),
        output_tokens=max(0, completion - reasoning),
        reasoning_tokens=reasoning,
        cache_read_tokens=cache_read,
    )

    rec = CallRecord(
        provider=str(_first(row, "custom_llm_provider", "provider", default="openai")),
        model=str(model),
        route=str(_first(row, "call_type", "route", default="/chat/completions")),
        user_id=str(_first(row, "user", "end_user", "user_id",
                           default=metadata.get("user_api_key_user_id") or "unknown")),
        session_id=str(_first(row, "session_id", "request_id", "litellm_call_id",
                              default=metadata.get("session_id") or "unknown")),
        ts=str(_first(row, "startTime", "start_time", "timestamp", default="")),
        reported=usage,
        request_text=_text_from(_first(row, "messages", "request", "proxy_server_request")),
        response_text=_text_from(_first(row, "response", "completion", "response_text")),
        # The provider's OWN reported $ — LiteLLM logs it as `spend` / `response_cost`. The actual
        # charge; read it instead of asking anyone to type rates.
        reported_cost_usd=_to_float(_first(row, "spend", "response_cost", "cost")),
    )
    # Activity type: explicit tag/metadata first, else local heuristic on the request text.
    from ..classify import classify_call
    rec.task_class = classify_call(rec, metadata)
    return rec


def ingest_litellm_spendlog(path: str, store: Store) -> int:
    """Ingest a LiteLLM SpendLogs JSONL file into the store. Returns count ingested."""
    n = 0
    for row in _iter_jsonl(path):
        rec = parse_litellm_row(row)
        if rec is not None:
            store.record(rec)
            n += 1
    return n


def _iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return
    # Accept either JSONL (one object per line) or a single JSON array.
    if content[0] == "[":
        for row in json.loads(content):
            if isinstance(row, dict):
                yield row
        return
    for line in content.splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)
