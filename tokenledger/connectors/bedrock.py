"""
AWS Bedrock connector (SKELETON — built against AWS's PUBLISHED model-invocation-logging schema,
NOT yet against a real partner's logs). Reads Bedrock "ModelInvocationLog" records (delivered to
CloudWatch Logs or S3 when model invocation logging is enabled) and normalises each into a
TokenLedger CallRecord.

  Docs: AWS Bedrock "Monitor model invocation using CloudWatch Logs and Amazon S3".

HONESTY / STATUS (read before trusting field mappings):
  The TOP-LEVEL shape (schemaType/timestamp/modelId/operation/input.inputTokenCount/
  output.outputTokenCount) is stable and documented. The BODY shape (inputBodyJson/outputBodyJson)
  and the Converse-API usage block vary by model family and API, and the body is only present when
  "log model input/output data" is enabled (large bodies go to S3 via inputBodyS3Path/
  outputBodyS3Path, which we record but do NOT fetch — no egress). Every place where the exact JSON
  path is model/API-specific is marked `# TODO(partner): confirm against real logs`. Treat token
  TEXT extraction as best-effort: if we can't recover the text, output stays UNVERIFIABLE (the
  reconciler labels it honestly) rather than guessed.

No data egress: this only reads a local file. S3 body pointers are recorded, never fetched.
"""

from __future__ import annotations

import json
from typing import Any, Iterator, Optional

from ..core import CallRecord, Usage
from ..store import Store

# Cross-region inference profiles prefix the modelId with a region group (us./eu./apac.). Strip it
# to find the vendor; keep the FULL id as the model so tokenizer substring-routing still works.
_REGION_PREFIXES = ("us.", "eu.", "apac.", "us-gov.")


def _provider_model_from_bedrock_id(model_id: str) -> tuple[str, str]:
    """Map a Bedrock modelId to (provider, model) for tokenizer routing in core.count_tokens.

    e.g. 'anthropic.claude-3-sonnet-20240229-v1:0'    -> ('anthropic', <full id>)  [closed -> BOUNDED]
         'us.meta.llama3-1-70b-instruct-v1:0'         -> ('meta',      <full id>)  [open -> HF EXACT]
         'amazon.nova-pro-v1:0'                        -> ('amazon',    <full id>)  [closed -> BOUNDED]

    The FULL id is kept as the model so core's open-weight substring map (llama/mistral/...) still
    fires; provider drives the closed-vs-open decision. anthropic/amazon/cohere/ai21 have no public
    tokenizer -> core estimates (BOUNDED), which is the honest outcome."""
    mid = model_id or ""
    vendor_src = mid
    for p in _REGION_PREFIXES:
        if vendor_src.startswith(p):
            vendor_src = vendor_src[len(p):]
            break
    provider = vendor_src.split(".", 1)[0] if "." in vendor_src else (vendor_src or "bedrock")
    return provider, mid


def _text_from_bedrock_body(body: Any, *, is_output: bool) -> str:
    """Best-effort recovery of prompt/response TEXT from a Bedrock body for re-tokenisation. Handles
    the common InvokeModel (Anthropic/Titan) and Converse shapes; returns '' when it can't (output
    then stays UNVERIFIABLE — never guessed). # TODO(partner): confirm/extend per model family."""
    if body is None:
        return ""
    if isinstance(body, str):
        return body
    if not isinstance(body, dict):
        return str(body)

    parts: list[str] = []

    def _content_blocks(blocks: Any) -> None:
        # Converse/Anthropic content: list of {"text": "..."} (or other block types we skip).
        if isinstance(blocks, list):
            for b in blocks:
                if isinstance(b, dict) and isinstance(b.get("text"), str):
                    parts.append(b["text"])
                elif isinstance(b, str):
                    parts.append(b)
        elif isinstance(blocks, str):
            parts.append(blocks)

    if is_output:
        # Anthropic InvokeModel: {"content":[{"type":"text","text":...}]} or {"completion": "..."}
        if "content" in body:
            _content_blocks(body.get("content"))
        if isinstance(body.get("completion"), str):
            parts.append(body["completion"])
        # Converse: {"output":{"message":{"content":[{"text":...}]}}}
        out = body.get("output")
        if isinstance(out, dict):
            msg = out.get("message", {})
            if isinstance(msg, dict):
                _content_blocks(msg.get("content"))
        # Titan: {"results":[{"outputText": "..."}]}
        for r in (body.get("results") or []):
            if isinstance(r, dict) and isinstance(r.get("outputText"), str):
                parts.append(r["outputText"])
    else:
        # Anthropic/Converse input: {"messages":[{"role","content": <str|blocks>}]}
        for m in (body.get("messages") or []):
            if isinstance(m, dict):
                _content_blocks(m.get("content"))
        # Anthropic legacy / Titan: {"prompt": "..."} / {"inputText": "..."}
        for k in ("prompt", "inputText", "system"):
            v = body.get(k)
            if isinstance(v, str):
                parts.append(v)
    return "\n".join(p for p in parts if p)


def _usage_from(record: dict, output: dict) -> Usage:
    """Read token counts defensively. Prefer the documented top-level input.inputTokenCount /
    output.outputTokenCount; fall back to a Converse usage block in the output body. Cache fields
    appear under a few names across versions. # TODO(partner): confirm cache field names on real logs."""
    inp = record.get("input")
    inp = inp if isinstance(inp, dict) else {}
    in_ct = inp.get("inputTokenCount")
    out_ct = output.get("outputTokenCount")

    # Converse fallback: output.outputBodyJson.usage.{inputTokens,outputTokens,...}
    usage_block = {}
    body = output.get("outputBodyJson")
    if isinstance(body, dict) and isinstance(body.get("usage"), dict):
        usage_block = body["usage"]
    if in_ct is None:
        in_ct = usage_block.get("inputTokens")
    if out_ct is None:
        out_ct = usage_block.get("outputTokens")

    cache_read = (inp.get("cacheReadInputTokenCount")
                  or output.get("cacheReadInputTokenCount")
                  or usage_block.get("cacheReadInputTokens") or 0)
    cache_creation = (inp.get("cacheWriteInputTokenCount")
                      or output.get("cacheWriteInputTokenCount")
                      or usage_block.get("cacheWriteInputTokens") or 0)

    return Usage(
        input_tokens=int(in_ct or 0),
        output_tokens=int(out_ct or 0),
        reasoning_tokens=0,   # Bedrock invocation logs do not expose a separate reasoning count
        cache_read_tokens=int(cache_read or 0),
        cache_creation_tokens=int(cache_creation or 0),
    )


def parse_bedrock_record(record: dict) -> Optional[CallRecord]:
    """Map one Bedrock ModelInvocationLog dict to a CallRecord. Returns None if it isn't usable."""
    if not isinstance(record, dict):
        return None
    model_id = record.get("modelId")
    if not model_id:
        return None
    inp = record.get("input")
    inp = inp if isinstance(inp, dict) else {}        # tolerate a malformed/non-dict field
    output = record.get("output")
    output = output if isinstance(output, dict) else {}

    provider, model = _provider_model_from_bedrock_id(str(model_id))
    usage = _usage_from(record, output)

    request_text = _text_from_bedrock_body(inp.get("inputBodyJson"), is_output=False)
    response_text = _text_from_bedrock_body(output.get("outputBodyJson"), is_output=True)

    identity = record.get("identity") or {}
    rec = CallRecord(
        provider=provider,
        model=model,
        route=str(record.get("operation", "InvokeModel")),
        # No per-user field in the base schema; the IAM identity ARN is the closest stable actor.
        # TODO(partner): map to a real user/team via tags or a metadata convention.
        user_id=str(identity.get("arn", "unknown")),
        session_id=str(record.get("requestId", "unknown")),
        ts=str(record.get("timestamp", "")),
        reported=usage,
        request_text=request_text,
        response_text=response_text,
    )
    from ..classify import classify_call
    rec.task_class = classify_call(rec, {})
    return rec


def ingest_bedrock_invocation_logs(path: str, store: Store) -> int:
    """Ingest a Bedrock invocation-log export (JSONL — one record per line — or a JSON array) into
    the store. Returns the count ingested. CloudWatch Logs exports are commonly JSONL; an S3
    delivery may be an array. Records without a modelId are skipped."""
    n = 0
    for rec_dict in _iter_records(path):
        rec = parse_bedrock_record(rec_dict)
        if rec is not None:
            store.record(rec)
            n += 1
    return n


def _iter_records(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return
    if content[0] == "[":
        for row in json.loads(content):
            if isinstance(row, dict):
                yield row
        return
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue  # one corrupt line must not abort a large export (skip the record)
        # CloudWatch Logs may wrap the event as {"message": "<json string>"} — unwrap if so.
        if isinstance(obj, dict) and "modelId" not in obj and isinstance(obj.get("message"), str):
            try:
                obj = json.loads(obj["message"])
            except Exception:
                continue
        if isinstance(obj, dict):
            yield obj


# --- Bedrock model-swap matrix --------------------------------------------------------
#
# Premium in-cloud model -> cheaper in-cloud CANDIDATE(s) (keeps the committed spend, security and
# data inside the customer's AWS account — the "switch the model inside your cloud" wedge).
# These are CANDIDATES from public positioning ONLY; the actual quality delta MUST come from the
# customer's own evals (the evaluator never asserts a switch without measured quality). Keyed by a
# substring of the modelId.
# TODO(partner): confirm exact current SKUs/ids and refresh as the Bedrock catalogue changes.
BEDROCK_MODEL_SWAP_MATRIX: dict[str, list[str]] = {
    "claude-3-opus":     ["amazon.nova-pro-v1:0", "meta.llama3-1-70b-instruct-v1:0"],
    "claude-3-5-sonnet": ["amazon.nova-pro-v1:0", "meta.llama3-1-70b-instruct-v1:0"],
    "claude-3-7-sonnet": ["amazon.nova-pro-v1:0", "meta.llama3-1-70b-instruct-v1:0"],
    "claude-3-sonnet":   ["amazon.nova-lite-v1:0", "meta.llama3-1-8b-instruct-v1:0"],
    "claude-3-haiku":    ["amazon.nova-micro-v1:0"],
    "llama3-1-70b":      ["amazon.nova-lite-v1:0"],
}


def bedrock_model_swap_candidates(model_id: str) -> list[str]:
    """Cheaper in-cloud candidate model ids for a premium Bedrock model, from the public matrix.
    Returns [] when there is no suggestion. These are STARTING POINTS for an evaluation — the
    customer's evals decide; we never claim a swap is safe on public benchmarks alone."""
    mid = (model_id or "").lower()
    for key, cands in BEDROCK_MODEL_SWAP_MATRIX.items():
        if key in mid:
            return list(cands)
    return []
