"""Pre-built connectors that normalise provider/gateway logs into TokenLedger CallRecords.

Connectors are the onboarding moat for the product-led motion (Metronome lesson): a team
points TokenLedger at the logs their gateway already produces and is running in minutes, with
no custom glue. Every connector normalises INTO the one canonical CallRecord schema.
"""

from .litellm import ingest_litellm_spendlog, parse_litellm_row
from .llm_bridge import MeteringLLMClient, metered
from .bedrock import (
    ingest_bedrock_invocation_logs, parse_bedrock_record,
    bedrock_model_swap_candidates, BEDROCK_MODEL_SWAP_MATRIX,
)

__all__ = [
    "ingest_litellm_spendlog",
    "parse_litellm_row",
    "MeteringLLMClient",
    "metered",
    "ingest_bedrock_invocation_logs",
    "parse_bedrock_record",
    "bedrock_model_swap_candidates",
    "BEDROCK_MODEL_SWAP_MATRIX",
]
