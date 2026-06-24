"""
Offline demo. No API key needed. The output-token re-count is REAL (tiktoken if installed,
estimator otherwise); the "reported" usage is synthetic so we can plant discrepancies and
show the engine catch them.

Run:  python -m examples.demo     (from the repo root)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenledger import (  # noqa: E402
    Store, Usage, QualitySignal, record_call, print_summary, write_html,
)
from tokenledger.core import count_tokens, reconcile_billing_period  # noqa: E402
from tokenledger.dashboard import discrepancy_report_md  # noqa: E402

ANSWER = (
    "The mitochondria is the powerhouse of the cell. It generates most of the cell's "
    "supply of adenosine triphosphate, used as a source of chemical energy. Beyond ATP "
    "production it is involved in signalling, cellular differentiation, and cell death, "
    "as well as maintaining control of the cell cycle and cell growth."
)
PROMPT = "Explain in a short paragraph what the mitochondria does in a cell."


def main() -> None:
    db = "tokenledger_demo.db"
    if os.path.exists(db):
        os.remove(db)
    store = Store(db)

    # Honest count of the answer text, so synthetic "reported" values are realistic.
    true_out, conf = count_tokens(ANSWER, "openai", "gpt-4o")
    true_in, _ = count_tokens(PROMPT, "openai", "gpt-4o")
    print(f"(re-tokenized answer = {true_out} output tokens, confidence={conf.value})")

    # 1. Clean OpenAI call — reported matches the text. Quality KNOWN AT CALL TIME (record-time).
    record_call(store, provider="openai", model="gpt-4o", user_id="alice",
                session_id="s-001", ts="2026-06-20T10:00:00Z", task_class="coding",
                usage=Usage(input_tokens=true_in + 6, output_tokens=true_out),
                request_text=PROMPT, response_text=ANSWER,
                quality=QualitySignal(eval_score=0.92, status="accept", success=True))

    # 2. OpenAI call that OVER-bills output by 40% — should be caught exactly. We keep the id so a
    # DOWNSTREAM reviewer can set the quality later (the deferred path), as quality usually is.
    rec2 = record_call(store, provider="openai", model="gpt-4o", user_id="alice",
                       session_id="s-002", ts="2026-06-20T10:05:00Z", task_class="coding",
                       usage=Usage(input_tokens=true_in + 6, output_tokens=int(true_out * 1.4)),
                       request_text=PROMPT, response_text=ANSWER)

    # 3. OpenAI call with a wildly inflated INPUT count — out-of-band (bounded) flag.
    record_call(store, provider="openai", model="gpt-4o-mini", user_id="bob",
                session_id="s-003", ts="2026-06-20T10:10:00Z", task_class="summarisation",
                usage=Usage(input_tokens=true_in * 8, output_tokens=true_out),
                request_text=PROMPT, response_text=ANSWER,
                quality=QualitySignal(status="accept"))

    # 4. o1 reasoning call — most tokens are hidden reasoning, recorded as UNVERIFIABLE.
    record_call(store, provider="openai", model="o1", user_id="bob",
                session_id="s-004", ts="2026-06-20T10:15:00Z", task_class="coding",
                usage=Usage(input_tokens=true_in + 6, output_tokens=true_out,
                            reasoning_tokens=4200),
                request_text=PROMPT, response_text=ANSWER,
                quality=QualitySignal(status="reject", eval_score=0.3))

    # 5. Anthropic call — no exact tokenizer, so output is an estimate band; cache recorded.
    # No quality signal: it stays UNLABELED (None), excluded from cost-per-accepted, surfaced as
    # a count — the honest 'not captured' state, never silently read as a reject.
    record_call(store, provider="anthropic", model="claude-sonnet-4", user_id="alice",
                session_id="s-005", ts="2026-06-20T10:20:00Z", task_class="outreach",
                usage=Usage(input_tokens=true_in, output_tokens=true_out,
                            cache_read_tokens=1500),
                request_text=PROMPT, response_text=ANSWER, route="/v1/messages")

    # DEFERRED quality on call 2, by the natural key (session_id + request_sha) a downstream
    # eval/accept event would carry. This is how quality lands in practice — after the call.
    store.set_quality_by(rec2.session_id, rec2.request_sha,
                         QualitySignal(status="reject", eval_score=0.4, success=False))

    print_summary(store)

    # The third number: provider billing-API total vs the sum we captured.
    captured_sum = sum(r.reported.output_tokens for r in store.all_records())
    v, note = reconcile_billing_period(reported_total=captured_sum + 9000,
                                       per_call_sum=captured_sum)
    print(f"Billing-period reconciliation: {v.value} — {note}\n")

    html_path = write_html(store, "tokenledger_demo.html")
    with open("discrepancy_report.md", "w", encoding="utf-8") as f:
        f.write(discrepancy_report_md(store))
    print(f"Wrote {html_path} and discrepancy_report.md")


if __name__ == "__main__":
    main()
