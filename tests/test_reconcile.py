import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenledger.core import (  # noqa: E402
    CallRecord, Usage, Verdict, Confidence, reconcile_call,
    reconcile_billing_period, count_tokens, PRICING,
)

TEXT = "The quick brown fox jumps over the lazy dog. " * 5


def _rec(out_reported, provider="openai", model="gpt-4o", in_reported=20):
    return CallRecord(
        provider=provider, model=model, route="/chat/completions",
        user_id="u", session_id="s", ts="2026-06-20T00:00:00Z",
        reported=Usage(input_tokens=in_reported, output_tokens=out_reported),
        request_text="hi", response_text=TEXT,
    )


def test_output_overcount_caught_when_exact():
    true_out, conf = count_tokens(TEXT, "openai", "gpt-4o")
    rc = reconcile_call(_rec(out_reported=true_out + 50))
    out = next(b for b in rc.buckets if b.bucket == "output")
    if conf is Confidence.EXACT:
        assert out.verdict == Verdict.OVERCOUNT
        assert rc.has_overcount
    else:  # estimator environment: still flags as out of band
        assert out.verdict in (Verdict.OUT_OF_BAND, Verdict.OVERCOUNT)


def test_clean_output_passes():
    true_out, _ = count_tokens(TEXT, "openai", "gpt-4o")
    rc = reconcile_call(_rec(out_reported=true_out))
    out = next(b for b in rc.buckets if b.bucket == "output")
    assert out.verdict == Verdict.OK
    assert not rc.has_overcount


def test_input_out_of_band():
    rc = reconcile_call(_rec(out_reported=count_tokens(TEXT, "openai", "gpt-4o")[0],
                             in_reported=5000))
    inp = next(b for b in rc.buckets if b.bucket == "input")
    assert inp.verdict == Verdict.OUT_OF_BAND
    assert inp.confidence == Confidence.BOUNDED


def test_reasoning_unverifiable():
    rec = _rec(out_reported=count_tokens(TEXT, "openai", "gpt-4o")[0])
    rec.reported.reasoning_tokens = 3000
    rc = reconcile_call(rec)
    r = next(b for b in rc.buckets if b.bucket == "reasoning")
    assert r.confidence == Confidence.UNVERIFIABLE
    assert r.verdict == Verdict.UNCHECKABLE


def test_adapters_normalize_to_disjoint_no_double_count():
    # OpenAI reports OVERLAPPING (prompt_tokens INCL cached; completion_tokens INCL reasoning).
    # from_openai must SUBTRACT the subsets so each token is counted ONCE (else cost double-counts).
    from tokenledger.recorder import from_openai, from_anthropic
    from tokenledger.core import _cost
    oa = from_openai({"prompt_tokens": 1000, "completion_tokens": 1186,
                      "completion_tokens_details": {"reasoning_tokens": 1024},
                      "prompt_tokens_details": {"cached_tokens": 200}})
    # disjoint: input = 1000-200, output = 1186-1024, reasoning, cache_read
    assert (oa.input_tokens, oa.output_tokens, oa.reasoning_tokens, oa.cache_read_tokens) == (800, 162, 1024, 200)
    # cost counts each token once: 800*pin + 162*pout + 1024*preason + 200*pcache
    pin, pout, prn, pc = PRICING["gpt-4o"]
    assert abs(_cost("gpt-4o", oa) - (800*pin + 162*pout + 1024*prn + 200*pc) / 1e6) < 1e-12
    # Anthropic already reports DISJOINT (input = uncached, cache separate) -> passthrough
    an = from_anthropic({"input_tokens": 80, "output_tokens": 30,
                         "cache_read_input_tokens": 50, "cache_creation_input_tokens": 20})
    assert (an.input_tokens, an.output_tokens, an.cache_read_tokens, an.cache_creation_tokens) == (80, 30, 50, 20)
    print("PASS test_adapters_normalize_to_disjoint_no_double_count")


def test_normalized_usage_not_re_adapted():
    from tokenledger.store import Store
    from tokenledger.recorder import record_call
    import os
    db = "_t.db"
    if os.path.exists(db):
        os.remove(db)
    s = Store(db)
    record_call(s, provider="openai", model="o1", user_id="u", session_id="s",
                ts="2026-06-20T00:00:00Z",
                usage=Usage(input_tokens=10, output_tokens=5, reasoning_tokens=4200))
    rec = s.all_records()[0]
    assert rec.reported.reasoning_tokens == 4200  # must survive, not be dropped by adapter
    os.remove(db)


def test_missing_text_is_unverifiable_not_overcount():
    # A row with no captured response/request text must NOT be flagged as an over-count.
    rec = CallRecord(
        provider="openai", model="o1", route="/chat/completions",
        user_id="u", session_id="s", ts="2026-06-21T00:00:00Z",
        reported=Usage(input_tokens=5000, output_tokens=64, reasoning_tokens=4200),
        request_text="", response_text="",
    )
    rc = reconcile_call(rec)
    out = next(b for b in rc.buckets if b.bucket == "output")
    inp = next(b for b in rc.buckets if b.bucket == "input")
    assert out.verdict == Verdict.UNCHECKABLE and out.confidence == Confidence.UNVERIFIABLE
    assert inp.verdict == Verdict.UNCHECKABLE and inp.confidence == Confidence.UNVERIFIABLE
    assert not rc.has_overcount  # the whole point: no false positive


def test_activity_classification():
    from tokenledger.classify import classify_text, classify_from_tags
    assert classify_text("Please refactor this function and fix the traceback\n```py\n```") == "coding"
    assert classify_text("draft a cold email to follow up with the prospect") == "outreach"
    assert classify_text("write a press release announcement for the newsroom") == "pr"
    assert classify_text("summarise this email and add to the meeting agenda") in ("admin", "summarisation")
    # explicit tag wins (deterministic)
    assert classify_from_tags({"task_class": "coding"}) == "coding"
    assert classify_from_tags(["outreach"]) == "outreach"
    assert classify_from_tags({"use_case": "sales"}) == "outreach"  # alias on a known key


def test_conformance_catches_jitter_not_consistent_bias():
    from tokenledger.conformance import run_conformance
    true = lambda t: count_tokens(t, "openai", "gpt-4o")[0]  # noqa: E731
    assert run_conformance(true).passed                      # honest meter passes
    calls = {"n": 0}
    def jit(t):
        calls["n"] += 1
        return true(t) + (calls["n"] % 3)                    # non-deterministic
    assert not run_conformance(jit).passed                   # caught
    assert run_conformance(lambda t: int(round(true(t) * 1.1))).passed  # consistent bias slips (documented limit)


def test_calibration_catches_systematic_bias():
    from tokenledger.calibration import detect_systematic_bias, invariant_probes, validate_invariance
    tt = lambda t: count_tokens(t, "openai", "gpt-4o")[0]  # noqa: E731
    assert validate_invariance(invariant_probes(), {"tiktoken": tt})["tiktoken"]
    assert not detect_systematic_bias(tt).biased                      # faithful
    r = detect_systematic_bias(lambda t: int(round(tt(t) * 1.10)))    # the bias conformance missed
    assert r.biased and 1.07 < r.slope < 1.13                         # recovered ~+10%


def test_open_weight_model_not_mislabeled_tiktoken_under_openai_provider():
    # Regression guard for the count_tokens reorder: an open-weight model served behind an
    # OpenAI-COMPATIBLE endpoint (vLLM/Ollama/internal gateway, provider="openai") must be
    # counted with ITS OWN tokenizer — never tiktoken's count mislabeled EXACT. Without the
    # reorder, the tiktoken branch would fire first and return (tiktoken_count, EXACT).
    import tokenledger.core as core
    text = "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)  # recursion\n" * 4
    tk_count, tk_conf = count_tokens(text, "openai", "gpt-4o")   # genuine OpenAI → tiktoken EXACT
    assert tk_conf is Confidence.EXACT
    # Mistral's tokenizer gives a provably different count for this text than tiktoken (133 vs
    # 112) — so the count alone discriminates the new ordering from the old. (Some open-weight
    # tokenizers happen to agree with tiktoken on a given string; Mistral does not here.)
    cnt, conf = count_tokens(text, "openai", "mistral-7b")  # open-weight served via openai provider
    # Invariant (always holds): an open-weight model is NEVER given tiktoken's number as EXACT.
    assert not (conf is Confidence.EXACT and cnt == tk_count), \
        "open-weight model under an openai provider must not get tiktoken's count labeled EXACT"
    if core._HAVE_HF and conf is Confidence.EXACT:
        # Real tokenizer reachable → EXACT via HF, and it differs from tiktoken (proves the
        # reorder: tiktoken did NOT win for this open-weight model).
        assert cnt != tk_count
    else:
        # Tokenizer file unreachable (offline) → BOUNDED estimate, never a tiktoken EXACT count.
        assert conf is Confidence.BOUNDED


def test_reasoning_canonical_disjoint_no_false_overcount():
    # CANONICAL pipeline: from_openai normalises raw OVERLAPPING usage (completion incl reasoning) to
    # DISJOINT (visible output + separate reasoning), so reconcile compares visible output directly —
    # no false OVERCOUNT, and reasoning is NOT double-counted in cost.
    from tokenledger.recorder import from_openai
    visible = "The answer is 42."
    vis, _ = count_tokens(visible, "openai", "gpt-4o")
    u = from_openai({"prompt_tokens": 20, "completion_tokens": vis + 1024,
                     "completion_tokens_details": {"reasoning_tokens": 1024}})
    assert u.output_tokens == vis and u.reasoning_tokens == 1024   # disjoint
    rec = CallRecord(provider="openai", model="gpt-4o", route="/v1", user_id="u", session_id="s",
                     ts="2026-06-24T00:00:00Z", reported=u, request_text="q", response_text=visible)
    out = next(b for b in reconcile_call(rec).buckets if b.bucket == "output")
    assert out.verdict == Verdict.OK, f"canonical visible output must reconcile OK (got {out.verdict})"
    rb = next(b for b in reconcile_call(rec).buckets if b.bucket == "reasoning")
    assert rb.verdict == Verdict.UNCHECKABLE                       # reasoning judged separately
    # a REAL visible over-count is still caught
    rec.reported.output_tokens = vis + 50
    out2 = next(b for b in reconcile_call(rec).buckets if b.bucket == "output")
    assert out2.verdict == Verdict.OVERCOUNT, "a real visible over-count must still be caught"
    print("PASS test_reasoning_canonical_disjoint_no_false_overcount")


def test_closed_provider_small_output_not_false_flagged():
    # HARNESS-FOUND BUG (2026-06-23): a closed provider (anthropic -> BOUNDED estimate) with a SMALL
    # output must not false-flag OUT_OF_BAND for a normal figure — the multiplicative band is tiny
    # for small N, so the additive cushion is required (the input band always had it; output didn't).
    short = "ok, done."
    est, conf = count_tokens(short, "anthropic", "claude-sonnet-4")
    assert conf is Confidence.BOUNDED          # closed provider is estimate-only
    rec = CallRecord(provider="anthropic", model="claude-sonnet-4", route="/v1", user_id="u",
                     session_id="s", ts="2026-06-20T00:00:00Z",
                     reported=Usage(input_tokens=15, output_tokens=est + 20),  # normal wobble
                     request_text="hi", response_text=short)
    out = next(b for b in reconcile_call(rec).buckets if b.bucket == "output")
    assert out.verdict == Verdict.OK, f"small estimate wobble must NOT be OUT_OF_BAND (got {out.verdict})"
    # but a genuinely large over-count is STILL caught
    rec2 = CallRecord(provider="anthropic", model="claude-sonnet-4", route="/v1", user_id="u",
                      session_id="s", ts="2026-06-20T00:00:00Z",
                      reported=Usage(input_tokens=15, output_tokens=est + 500),
                      request_text="hi", response_text=short)
    out2 = next(b for b in reconcile_call(rec2).buckets if b.bucket == "output")
    assert out2.verdict == Verdict.OUT_OF_BAND, "a large over-count must still be flagged"
    print("PASS test_closed_provider_small_output_not_false_flagged")


def test_billing_period_mismatch():
    v, _ = reconcile_billing_period(reported_total=10000, per_call_sum=8000)
    assert v == Verdict.OVERCOUNT
    v2, _ = reconcile_billing_period(reported_total=8000, per_call_sum=8000)
    assert v2 == Verdict.OK


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
