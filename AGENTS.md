# AGENTS.md

Guidance for AI agents and automation using TokenLedger.

## What this tool is for
Independently re-count and reconcile the tokens an LLM provider bills, locally, with no data egress.
It verifies the provider's numbers rather than trusting them, and labels every figure by confidence.

## How to use it after a call
After any provider call, hand TokenLedger the raw usage object plus the text:

```python
from tokenledger import Store, record_call
store = Store("tokenledger.db")
record_call(store, provider="openai", model="gpt-4o",
            user_id=user, session_id=session, ts=iso_now,
            usage=response.usage,                 # raw OpenAI or Anthropic usage object
            request_text=serialized_prompt, response_text=completion_text)
```

Then read the reconciliation: `tokenledger report --db tokenledger.db --html report.html`.

## The honesty contract (do not break it)
- Every bucket is labelled EXACT, BOUNDED, or UNVERIFIABLE. Never present a BOUNDED estimate as an
  exact result, and never assert an UNVERIFIABLE figure (reasoning, per-call cache).
- Output is EXACT only where a public tokenizer exists (OpenAI, open-weight). Closed models such as
  Claude and Gemini are BOUNDED.
- Absence of a signal is never read as a negative. No captured text means UNVERIFIABLE, not an
  over-count. No quality label means unlabeled, not a reject.
- Cost-per-accepted is descriptive. It is not a migration recommendation.

## Exit and verdict semantics
`reconcile_call` returns per-bucket verdicts: `OK`, `OVERCOUNT`, `UNDERCOUNT`, `OUT_OF_BAND`, or
`UNCHECKABLE`. Treat `OVERCOUNT` and `OUT_OF_BAND` as discrepancies to investigate, not proof of
fraud. Confirm against the provider's own billing total before acting.

## What it does not do
It does not route or proxy production traffic by default, it does not call the provider's
count-tokens API (that would send your text out and would not be independent), and it does not make
any network call except the optional passthrough proxy forwarding your own request.
