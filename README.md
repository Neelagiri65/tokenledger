# TokenLedger

Self-hosted, cross-provider LLM token metering and reconciliation. It logs every billable API
call across providers and sessions, independently re-counts the text that is checkable, reconciles
the provider's numbers three ways, and produces a dashboard plus a discrepancy report you can take
to the provider. It sits on your own infrastructure. No prompt or response content leaves the box.

TokenLedger is by [Nativerse](https://nativerse-ventures.com). Landing page and walkthrough:
[tokenledger.nativerse-ventures.com](https://tokenledger.nativerse-ventures.com).

## Why

Providers self-report token usage and bill you on it. The numbers are unsigned and rarely checked.
Billing is now multi-bucket (input, cached input, output, reasoning), and on reasoning models most
of what you pay for is hidden tokens you never see. TokenLedger makes the part that is checkable
checked, and is honest about the part that is not.

## What it verifies, and how honestly

- **Output tokens, EXACT** on providers that share a public tokenizer (OpenAI, open-weight models
  via tiktoken or the model's own tokenizer). It re-tokenises the text you actually received. Billed
  above counted is a hard discrepancy.
- **Input tokens, BOUNDED.** It re-counts what you sent plus documented message overhead, and flags
  figures outside a tolerance band. It cannot reconstruct hidden server-side additions.
- **Reasoning tokens, UNVERIFIABLE.** Billed but not returned. Recorded, never asserted.
- **Cache hit/miss, UNVERIFIABLE per call.** Provider-internal. Recorded, verify behaviourally
  across many calls.
- **Billing period, three-way.** The sum of captured per-call usage against the provider's
  billing or usage-API total. When a provider's own two numbers disagree, no tokenizer is needed.

Every result carries its confidence label. The tool never claims proof it does not have.

## Architectural constraint test (non-negotiables, checked before building)

1. No data egress. All counting and reconciliation are local. Content can be stored hashed only
   (`Store(redact=True)`).
2. Honest confidence. Each bucket is EXACT, BOUNDED, or UNVERIFIABLE. No false certainty.
3. Passive. Logging never changes or breaks the real call. A logging failure is swallowed.
4. Multi-provider, multi-session, multi-user. Pluggable adapters, every record tagged.
5. Vendor-neutral. Pricing and tolerances are configuration, not hard-coded assumptions.

## Quick start

```bash
pip install "tokenledger[exact]"        # the CLI plus tiktoken and tokenizers (exact mode)

tokenledger demo                         # offline demo: plants discrepancies, catches them
open tokenledger_demo.html               # the dashboard
```

`pip install tokenledger` alone runs in estimator mode, where exact-only buckets are labelled
BOUNDED instead of EXACT. Tokenisation runs locally. Only the public tokenizer file is fetched
once, and you can bundle it for air-gapped sites.

### Sidecar over an existing LiteLLM gateway

LiteLLM already writes spend logs. Point TokenLedger at them and it audits the numbers from the
outside. It is an out-of-band audit layer: it does not route or proxy your traffic, so it adds no
latency and no point of failure.

```bash
tokenledger ingest litellm_spendlogs.jsonl --format litellm --db tokenledger.db
tokenledger report --db tokenledger.db --html report.html --md discrepancy.md
open report.html
```

Enable `STORE_PROMPTS_IN_SPEND_LOGS=true` on LiteLLM so output tokens can be re-counted exactly.
Without captured text, output and input are reported as UNVERIFIABLE, never falsely flagged. Unlike
LiteLLM and Helicone callbacks, which hand back the provider's own reported usage, TokenLedger
re-tokenises independently. It verifies rather than aggregates.

### Docker

```bash
docker compose run --rm tokenledger demo
docker compose run --rm tokenledger ingest litellm_spendlogs.jsonl --format litellm
docker compose run --rm tokenledger report --html report.html
```

### Pre-built connectors

`tokenledger/connectors/` normalises gateway and provider logs into one canonical CallRecord
schema. LiteLLM ships today (`--format litellm`), native event JSONL via `--format jsonl`, and a
Bedrock invocation-log parser. Adapters normalise overlapping provider buckets to a disjoint model
so cost is never double-counted.

## Quality capture: cost per accepted output

Cost is half the picture. What did the spend buy? Attach an optional per-call quality signal
alongside cost, then read cost-per-accepted-output by activity in the CLI and HTML dashboard.

```python
from tokenledger import Store, record_call, QualitySignal
store = Store("tokenledger.db")
rec = record_call(store, provider="openai", model="gpt-4o", user_id=u, session_id=s,
                  ts=iso_now, usage=response.usage, request_text=prompt, response_text=completion)
# after your eval or accept event fires downstream:
store.set_quality_by(rec.session_id, rec.request_sha,
                     QualitySignal(eval_score=0.92, status="accept", success=True))
```

*Accepted* is driven by one signal only, `status == 'accept'`. Cost-per-accepted charges the cost
of rejected calls against the accepted ones, which is the waste signal. Calls with no label are
excluded and reported as a separate unlabeled count, never read as a reject. Cost-per-accepted is
descriptive. The thesis that it should drive model switching is not yet validated, and this view is
not a migration recommendation.

## Roadmap

1. **Reconciliation.** Catch billing discrepancies and quantify them in dollars.
2. **Quality capture.** Attach an optional task-level quality signal per call, record-time or
   deferred, and read cost-per-accepted-output by activity. Descriptive only.
3. **Model-switch advisory.** With cost and quality logged per task class, recommend moving
   workloads to cheaper or open-weight models where they match quality at lower cost. Staged as
   shadow, canary, then migrate, with the telemetry proving the call was right.

The migration thesis, when open-weight models are viable substitutes and at what cost-quality
trade-off, is being validated separately.

## Status

The reconciliation engine, store, recorder, dashboard, discrepancy report, pluggable cost model,
and migration evaluator are working and covered by the test suite. Real provider adapters parse
OpenAI, Anthropic, LiteLLM, and Bedrock usage shapes. Demand and the closed-model band width are
being validated with design partners. We do not claim a result we have not measured.

## Licence

Apache-2.0. See `LICENSE`.
