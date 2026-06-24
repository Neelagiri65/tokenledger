"""
Architectural constraint test for the pluggable cost model (written BEFORE the code,
per CLAUDE.md failure #1). These are the non-negotiable invariants:

  1. per_token cost is EXACT (config rate x a counted number) and equals the legacy
     per-call arithmetic — existing behaviour must not regress.
  2. flat_subscription and rented_compute costs are BOUNDED (depend on a utilisation/
     throughput assumption) and are NOT per_token_billable.
  3. The common denominator for ALL shapes is effective $/token = period_cost / measured
     tokens — this is what makes a rented open-weight model comparable to a per-token one.
  4. Cost confidence is ORTHOGONAL to count confidence (separate field).
  5. The overbill USD reconciliation is N/A for non-per_token_billable models — there is
     no per-token bill to dispute a dollar figure against (the silent-bug guard).
  6. With no registered cost model, EVERY model resolves to per_token from PRICING
     (zero behaviour change for existing data).
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retoken import core
from retoken.core import (
    Usage, Confidence, Verdict, CallRecord,
    PerTokenCost, FlatSubscriptionCost, RentedComputeCost,
    UsageAggregate, CostResult, cost_model_for, COST_MODELS, PRICING, _cost,
)


# --- 1. per_token: EXACT, equals legacy arithmetic --------------------------------------

def test_per_token_exact_and_matches_legacy():
    u = Usage(input_tokens=1000, output_tokens=500, reasoning_tokens=200, cache_read_tokens=100)
    cm = cost_model_for("gpt-4o")
    assert isinstance(cm, PerTokenCost)
    r = cm.call_cost(u)
    assert r.cost_confidence is Confidence.EXACT
    assert r.per_token_billable is True
    pin, pout, prn, pc = PRICING["gpt-4o"]
    expect = (1000 * pin + 500 * pout + 200 * prn + 100 * pc) / 1e6
    assert abs(r.usd - expect) < 1e-12
    # legacy _cost path is byte-identical
    assert abs(_cost("gpt-4o", u) - expect) < 1e-12
    print("PASS test_per_token_exact_and_matches_legacy")


def test_per_token_period_equals_sum_of_calls():
    # Linearity: period_cost over an aggregate == sum of per-call costs (EXACT).
    cm = cost_model_for("gpt-4o")
    calls = [Usage(100, 50, 0, 0), Usage(200, 80, 10, 5), Usage(0, 0, 0, 0)]
    per_call = sum(cm.call_cost(u).usd for u in calls)
    agg = UsageAggregate()
    for u in calls:
        agg.add_usage(u)
    period = cm.period_cost(agg)
    assert period.cost_confidence is Confidence.EXACT
    assert abs(period.usd - per_call) < 1e-12
    print("PASS test_per_token_period_equals_sum_of_calls")


# --- 2/3/4. flat + rented: BOUNDED, not billable, effective $/token ----------------------

def test_flat_subscription_bounded_and_effective_per_token():
    cm = FlatSubscriptionCost(fee_per_period=2000.0)
    assert cm.per_token_billable is False
    agg = UsageAggregate(input_tokens=8_000_000, output_tokens=2_000_000)  # 10M measured tokens
    r = cm.period_cost(agg)
    assert r.cost_confidence is Confidence.BOUNDED
    assert r.per_token_billable is False
    assert abs(r.usd - 2000.0) < 1e-9                      # period cost = the fixed fee
    assert abs(r.effective_per_token - 2000.0 / 10_000_000) < 1e-15
    # no measured tokens -> effective $/token undefined, never a divide-by-zero
    r0 = cm.period_cost(UsageAggregate())
    assert r0.effective_per_token is None
    print("PASS test_flat_subscription_bounded_and_effective_per_token")


def test_rented_compute_provisioned_hours():
    # 2 GPUs x $3.50/hr x 720h provisioned = $5040 for the period, amortised over tokens served.
    cm = RentedComputeCost(usd_per_gpu_hour=3.50, gpu_count=2, gpu_hours=720.0)
    assert cm.per_token_billable is False
    agg = UsageAggregate(input_tokens=400_000_000, output_tokens=100_000_000)  # 500M tokens
    r = cm.period_cost(agg)
    assert r.cost_confidence is Confidence.BOUNDED
    expect_period = 3.50 * 2 * 720.0
    assert abs(r.usd - expect_period) < 1e-9
    assert abs(r.effective_per_token - expect_period / 500_000_000) < 1e-15
    print("PASS test_rented_compute_provisioned_hours")


def test_rented_compute_throughput_fallback():
    # Hours unknown: derive effective $/token from $/GPU-hr ÷ throughput (assumed rate).
    cm = RentedComputeCost(usd_per_gpu_hour=4.0, gpu_count=1, throughput_tokens_per_hour=2_000_000)
    agg = UsageAggregate(output_tokens=10_000_000)
    r = cm.period_cost(agg)
    assert r.cost_confidence is Confidence.BOUNDED
    expect_eff = 4.0 * 1 / 2_000_000                       # $/token
    assert abs(r.effective_per_token - expect_eff) < 1e-15
    assert abs(r.usd - expect_eff * 10_000_000) < 1e-9
    print("PASS test_rented_compute_throughput_fallback")


def test_effective_per_token_is_period_cost_over_tokens_for_all_shapes():
    # The unifying invariant the migration evaluator relies on.
    agg = UsageAggregate(input_tokens=3_000_000, output_tokens=1_000_000)  # 4M tokens
    models = [
        PerTokenCost(2.0, 6.0, 6.0, 0.2),
        FlatSubscriptionCost(fee_per_period=500.0),
        RentedComputeCost(usd_per_gpu_hour=2.0, gpu_count=4, gpu_hours=100.0),
    ]
    for cm in models:
        r = cm.period_cost(agg)
        assert r.effective_per_token is not None
        assert abs(r.effective_per_token - r.usd / agg.total_tokens) < 1e-15
    print("PASS test_effective_per_token_is_period_cost_over_tokens_for_all_shapes")


# --- 5. overbill USD is N/A for non-per_token_billable (the silent-bug guard) ------------

def test_overbill_usd_na_for_capacity_models():
    from retoken.dashboard import _price
    assert _price("gpt-4o") is not None                    # per_token: a rate exists to dispute against
    COST_MODELS["rented-llama"] = RentedComputeCost(usd_per_gpu_hour=3.0, gpu_count=1, gpu_hours=10.0)
    try:
        assert _price("rented-llama") is None              # capacity: no per-token bill -> no $ dispute
    finally:
        del COST_MODELS["rented-llama"]
    print("PASS test_overbill_usd_na_for_capacity_models")


def test_capacity_overcount_tracks_tokens_but_not_dollars():
    # An output over-count on a capacity model records the token surplus but leaves overbill $ at 0
    # (there is no per-token bill to put a dollar figure on). Per-token models still get the $.
    from retoken.store import Store
    from retoken.dashboard import rollup_by, reconcile_all

    COST_MODELS["rented-qwen"] = RentedComputeCost(usd_per_gpu_hour=2.0, gpu_count=1, gpu_hours=24.0)
    fd, dbpath = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(dbpath)
    db = Store(dbpath)
    try:
        # response text "a b c" re-tokenises small; report a wildly inflated output to force overcount
        rec = CallRecord(
            provider="openai", model="rented-qwen", route="/v1/chat", user_id="u", session_id="s",
            ts="2026-06-23T00:00:00Z", reported=Usage(input_tokens=10, output_tokens=9999),
            request_text="hello", response_text="a b c",
        )
        db.record(rec)
        recs = reconcile_all(db)
        ru = rollup_by(recs, "provider")["openai"]
        assert ru.billed_usd > 0                            # period cost shown, not a silent 0
        assert ru.overbilled_usd == 0.0                     # no per-token bill -> no dollar dispute
    finally:
        del COST_MODELS["rented-qwen"]
        if os.path.exists(dbpath):
            os.remove(dbpath)
    print("PASS test_capacity_overcount_tracks_tokens_but_not_dollars")


def test_capacity_fee_amortised_across_buckets_not_multiplied():
    # A flat fee split across MULTIPLE rollup buckets must sum to the fee ONCE (amortised by token
    # share), not re-applied per bucket. This is the partition-invariance bug the per-bucket
    # period_cost() had — it needs >=2 buckets to surface.
    from retoken.store import Store
    from retoken.dashboard import rollup_by, reconcile_all, cost_per_accepted
    from retoken.quality import QualitySignal

    COST_MODELS["flat-model"] = FlatSubscriptionCost(fee_per_period=100.0)
    fd, dbpath = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(dbpath)
    db = Store(dbpath)
    try:
        # two equal-token calls, different task_class buckets; one accepted+labeled, one unlabeled.
        common = dict(provider="rented", model="flat-model", route="/v1", user_id="u",
                      ts="2026-06-23T00:00:00Z", reported=Usage(input_tokens=500, output_tokens=500))
        db.record(CallRecord(session_id="s1", task_class="coding",
                             quality=QualitySignal(status="accept"), **common))
        db.record(CallRecord(session_id="s2", task_class="admin", **common))  # unlabeled
        recs = reconcile_all(db)

        by_act = rollup_by(recs, "task_class")
        total_billed = sum(ru.billed_usd for ru in by_act.values())
        assert abs(total_billed - 100.0) < 1e-9, f"fee multiplied across buckets: {total_billed}"
        # each equal-token bucket carries half the fee
        assert abs(by_act["coding"].billed_usd - 50.0) < 1e-9
        assert abs(by_act["admin"].billed_usd - 50.0) < 1e-9
        # cost-per-accepted = labeled token share (the one accepted call), not the whole fee
        assert abs(by_act["coding"].labeled_billed_usd - 50.0) < 1e-9
        assert abs(cost_per_accepted(by_act["coding"]) - 50.0) < 1e-9

        # partition-invariance: the same period costs the same under a different rollup key
        by_prov = rollup_by(recs, "provider")
        assert abs(sum(ru.billed_usd for ru in by_prov.values()) - 100.0) < 1e-9
    finally:
        del COST_MODELS["flat-model"]
        if os.path.exists(dbpath):
            os.remove(dbpath)
    print("PASS test_capacity_fee_amortised_across_buckets_not_multiplied")


# --- 6. fallback + registry override ----------------------------------------------------

def test_cache_creation_rate_optional_default_zero_and_configurable():
    # cache-WRITE (creation) is an optional rate: built-in 4-tuple PRICING -> default 0 (conservative,
    # not charged); set it via config/PerTokenCost to cost it. Never double-counted with other buckets.
    u = Usage(input_tokens=100, output_tokens=50, cache_creation_tokens=1000)
    # built-in model: cache_creation defaults 0 -> only input+output charged
    builtin = cost_model_for("gpt-4o").call_cost(u).usd
    pin, pout, prn, pc = PRICING["gpt-4o"]
    assert abs(builtin - (100 * pin + 50 * pout) / 1e6) < 1e-12   # cache_creation contributes 0
    # configured cache-write rate (e.g. Anthropic ~1.25x input) IS charged, once
    cm = PerTokenCost(3.0, 15.0, 15.0, 0.3, cache_creation=3.75)
    got = cm.call_cost(u).usd
    assert abs(got - (100 * 3.0 + 50 * 15.0 + 1000 * 3.75) / 1e6) < 1e-12
    print("PASS test_cache_creation_rate_optional_default_zero_and_configurable")


def test_out_of_band_estimate_not_dollarised_as_exact_overbill():
    # HARNESS cross-cutting catch: a closed-provider OUT_OF_BAND (BOUNDED estimate) output must NOT
    # be charged into the precise 'overbill $' column — only an EXACT OVERCOUNT may. It is still
    # FLAGGED (a real discrepancy) but carries no exact dollar figure.
    from retoken.store import Store
    from retoken.dashboard import rollup_by, reconcile_all
    fd, dbpath = tempfile.mkstemp(suffix=".db"); os.close(fd); os.remove(dbpath)
    db = Store(dbpath)
    try:
        # anthropic (closed -> BOUNDED estimate); short response, hugely inflated reported output
        db.record(CallRecord(provider="anthropic", model="claude-sonnet-4", route="/v1", user_id="u",
                             session_id="s", ts="2026-06-23T00:00:00Z",
                             reported=Usage(input_tokens=10, output_tokens=400),
                             request_text="hi", response_text="a b c"))
        ru = rollup_by(reconcile_all(db), "provider")["anthropic"]
        assert ru.flagged == 1                     # still surfaced as a discrepancy
        assert ru.overbilled_usd == 0.0            # but NOT a precise dollar dispute (it's an estimate)
        assert ru.overbilled_output_tokens == 0
    finally:
        if os.path.exists(dbpath):
            os.remove(dbpath)
    print("PASS test_out_of_band_estimate_not_dollarised_as_exact_overbill")


def test_unregistered_model_falls_back_to_per_token_from_pricing():
    assert "totally-unknown-model-xyz" not in COST_MODELS
    cm = cost_model_for("totally-unknown-model-xyz")
    assert isinstance(cm, PerTokenCost)
    di, do, dr, dc = PRICING["_default"]
    assert (cm.input, cm.output, cm.reasoning, cm.cache_read) == (di, do, dr, dc)
    print("PASS test_unregistered_model_falls_back_to_per_token_from_pricing")


def test_registry_override_wins_over_pricing():
    COST_MODELS["gpt-4o"] = FlatSubscriptionCost(fee_per_period=1.0)
    try:
        cm = cost_model_for("gpt-4o")
        assert isinstance(cm, FlatSubscriptionCost)
        # per-call legacy cost collapses to 0 for a capacity model (no standalone marginal cost)
        assert _cost("gpt-4o", Usage(1000, 1000, 0, 0)) == 0.0
    finally:
        del COST_MODELS["gpt-4o"]
    # restored: back to per_token
    assert isinstance(cost_model_for("gpt-4o"), PerTokenCost)
    print("PASS test_registry_override_wins_over_pricing")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all tests passed")
