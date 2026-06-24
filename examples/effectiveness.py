"""
Effectiveness harness. Builds a LABELLED set of calls (we know the truth for each), runs
TokenLedger's reconciliation, and scores it: did it catch the real over-counts, avoid false
positives on clean calls, and correctly mark unverifiable ones? The output-token re-count is
REAL (tiktoken); only the "reported" numbers are synthetic so we can set ground truth.

Run:  python -m examples.effectiveness   (from repo root)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retoken.core import CallRecord, Usage, count_tokens, reconcile_call, Verdict  # noqa: E402

ANSWER = (
    "The mitochondria is the powerhouse of the cell. It generates most of the cell's supply of "
    "adenosine triphosphate, used as a source of chemical energy. Beyond ATP production it is "
    "involved in signalling, cellular differentiation, and cell death."
)
PROMPT = "Explain what the mitochondria does."

TRUE_OUT, _ = count_tokens(ANSWER, "openai", "gpt-4o")
TRUE_IN, _ = count_tokens(PROMPT, "openai", "gpt-4o")


def _rec(out_reported, *, label, in_reported=None, resp=ANSWER, req=PROMPT, provider="openai",
         model="gpt-4o"):
    return label, CallRecord(
        provider=provider, model=model, route="/chat/completions", user_id="u",
        session_id=label, ts="2026-06-21T00:00:00Z",
        reported=Usage(input_tokens=in_reported if in_reported is not None else TRUE_IN + 4,
                       output_tokens=out_reported),
        request_text=req, response_text=resp,
    )


# Labelled cases. should_flag = ground truth: is there a real discrepancy to catch?
CASES = [
    # clean calls — must NOT flag (tests false-positive rate)
    (*_rec(TRUE_OUT, label="clean-1"), False),
    (*_rec(TRUE_OUT, label="clean-2"), False),
    (*_rec(TRUE_OUT, label="clean-3"), False),
    # real over-counts — must flag (tests recall)
    (*_rec(int(TRUE_OUT * 1.4), label="overcount-40pct"), True),
    (*_rec(TRUE_OUT + 30, label="overcount-+30"), True),
    (*_rec(TRUE_OUT, in_reported=TRUE_IN * 10, label="input-inflated"), True),
    # unverifiable (no captured text) — must NOT flag (the trust-critical false-positive guard)
    (*_rec(9999, label="no-text", resp="", req=""), False),
]


def main() -> None:
    tp = fp = tn = fn = 0
    print(f"(ground-truth output re-count = {TRUE_OUT} tokens via tiktoken)\n")
    print(f"{'case':<20}{'truth':<10}{'detected':<10}{'result'}")
    print("-" * 55)
    for label, rec, should_flag in CASES:
        rc = reconcile_call(rec)
        detected = rc.has_overcount
        if should_flag and detected:
            res = "TP ✓"; tp += 1
        elif should_flag and not detected:
            res = "FN ✗ (missed)"; fn += 1
        elif not should_flag and detected:
            res = "FP ✗ (false alarm)"; fp += 1
        else:
            res = "TN ✓"; tn += 1
        print(f"{label:<20}{'flag' if should_flag else 'clean':<10}"
              f"{'flag' if detected else 'clean':<10}{res}")

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    print("-" * 55)
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"precision={precision:.0%}  recall={recall:.0%}  false-positive-rate="
          f"{fp / (fp + tn) if (fp + tn) else 0:.0%}")
    print("\nEffectiveness = catches every real discrepancy (recall) with zero false alarms "
          "(precision). For a trust product, FP=0 matters most.")


if __name__ == "__main__":
    main()
