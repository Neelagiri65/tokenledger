"""
Architectural constraint + acceptance tests for the migration evaluator
(retoken/evaluator.py). Written to the spec's NON-NEGOTIABLES:

  - No data egress / no network import in the module (test 7).
  - NEVER assert a switch without measured quality (tests 4, 5).
  - Cost confidence and quality confidence are SEPARATE fields — EXACT cost can coexist with
    unknown quality (tests 1, 4).
  - Passive/pure: evaluate_migration must not mutate the workload (test 8).

Runnable like the other suites: `python tests/test_evaluator.py` runs all and prints PASS lines.
Tolerates offline HF tokenizers (asserts EXACT-or-BOUNDED), but pins EXACT for an OpenAI
candidate when tiktoken is present.
"""
import copy
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retoken import core
from retoken.core import (
    CallRecord, Usage, Confidence, UsageAggregate,
    PerTokenCost, FlatSubscriptionCost, RentedComputeCost, PRICING, COST_MODELS,
    count_tokens, cost_model_for,
)
from retoken.quality import QualitySignal
from retoken.evaluator import (
    Candidate, CandidateProjection, MigrationReport,
    evaluate_migration, breakeven, to_dict,
    QUALITY_MEASURED, QUALITY_UNKNOWN,
)

try:
    import tiktoken  # noqa: F401
    _HAVE_TIKTOKEN = True
except Exception:
    _HAVE_TIKTOKEN = False


def _call(model, provider, req, resp, reported, **kw):
    return CallRecord(
        provider=provider, model=model, route="/v1/chat", user_id="u",
        session_id=kw.pop("session_id", "s1"), ts="2026-06-23T00:00:00Z",
        reported=reported, request_text=req, response_text=resp, **kw,
    )


# --- 1. per_token cheaper candidate: projected < current, EXACT cost when re-tokenised ----

def test_per_token_cheaper_candidate_exact_cost():
    # Clean input/output-only workload (NO reasoning/cache) so a per_token OpenAI candidate can be
    # EXACT (reasoning/cache buckets would otherwise force BOUNDED per the spec).
    req = "the quick brown fox jumps over the lazy dog and keeps running onward"
    resp = "a concise factual answer to the question that was asked here today"
    # Set reported == re-tokenised counts so current vs projected differ ONLY by RATE (no recount
    # noise) — makes savings_pct a real, hand-checkable number, not a tautology.
    in_ct, _ = count_tokens(req, "openai", "gpt-4o")
    out_ct, _ = count_tokens(resp, "openai", "gpt-4o")
    rec = _call("gpt-4o", "openai", req, resp,
                Usage(input_tokens=in_ct, output_tokens=out_ct))
    workload = [rec]

    cand = Candidate(model="gpt-4o-mini", provider="openai")  # cheaper per_token
    report = evaluate_migration(workload, [cand])
    proj = report.ranked[0]

    # current cost on gpt-4o, projected on gpt-4o-mini (same token counts, cheaper rates).
    pin0, pout0, _, _ = PRICING["gpt-4o"]
    pin1, pout1, _, _ = PRICING["gpt-4o-mini"]
    expect_current = (in_ct * pin0 + out_ct * pout0) / 1e6
    expect_proj = (in_ct * pin1 + out_ct * pout1) / 1e6
    assert abs(report.current_cost_usd - expect_current) < 1e-12
    assert abs(proj.projected_cost_usd - expect_proj) < 1e-12
    assert proj.projected_cost_usd < report.current_cost_usd
    expect_pct = (expect_current - expect_proj) / expect_current * 100.0
    assert abs(proj.savings_pct - expect_pct) < 1e-9

    # cost confidence: EXACT-or-BOUNDED in general; EXACT pinned when tiktoken is present (OpenAI).
    assert proj.cost_confidence in (Confidence.EXACT, Confidence.BOUNDED)
    if _HAVE_TIKTOKEN:
        assert proj.cost_confidence is Confidence.EXACT
    print("PASS test_per_token_cheaper_candidate_exact_cost")


# --- 2. capacity candidate: effective $/token, BOUNDED, projected period == sum proj tokens

def test_capacity_candidate_bounded_effective():
    req = "hello world this is a longer prompt to count"
    resp = "and here is a generated response of some length"
    rec = _call("gpt-4o", "openai", req, resp, Usage(input_tokens=100, output_tokens=80))
    workload = [rec]

    # Rented candidate (no HF tokenizer routing needed — provider blank, model arbitrary).
    cm = RentedComputeCost(usd_per_gpu_hour=3.0, gpu_count=1, gpu_hours=10.0)  # $30 period
    cand = Candidate(model="rented-endpoint", provider="", cost_model=cm)
    report = evaluate_migration(workload, [cand])
    proj = report.ranked[0]

    assert proj.cost_confidence is Confidence.BOUNDED
    # The projected period figure: effective $/token over the projected tokens x those tokens
    # recovers the full provisioned cost ($30) since the workload is the whole period here.
    agg = proj.projected_tokens
    assert isinstance(agg, UsageAggregate)
    period = cm.period_cost(agg)
    assert abs(proj.projected_cost_usd - period.usd) < 1e-9
    assert abs(proj.projected_cost_usd - 30.0) < 1e-9
    # projected period denominator == sum of projected tokens
    assert agg.total_tokens > 0
    print("PASS test_capacity_candidate_bounded_effective")


# --- 3. break-even: flat $100 vs baseline $2/1M -> V == 50,000,000 (hand calc) -------------

def test_breakeven_flat_vs_baseline_exact():
    flat = FlatSubscriptionCost(fee_per_period=100.0)
    b = 2.0 / 1e6                      # $2 per 1M tokens, expressed as $/token
    be = breakeven(flat, b)
    assert be.tokens == 50_000_000
    assert abs(be.usd_at_breakeven - 100.0) < 1e-12

    # rented with provisioned hours: fixed period cost behaves the same.
    rented = RentedComputeCost(usd_per_gpu_hour=3.0, gpu_count=1, gpu_hours=10.0)  # $30
    be_r = breakeven(rented, b)
    assert abs(be_r.tokens - 30.0 / b) < 1e-6

    # per_token vs per_token: linear, no break-even.
    be_lin = breakeven(PerTokenCost(2.0, 6.0, 6.0, 0.2), b)
    assert be_lin.tokens is None
    assert "linear" in be_lin.note.lower()
    print("PASS test_breakeven_flat_vs_baseline_exact")


# --- 4. no candidate quality -> unknown quality, cost_only, NOT safe, no "switch" ---------

def test_no_quality_cost_only_no_switch():
    req = "a clean prompt with only input and output buckets"
    resp = "a clean response without reasoning or cache tokens at all"
    in_ct, _ = count_tokens(req, "openai", "gpt-4o")
    out_ct, _ = count_tokens(resp, "openai", "gpt-4o")
    rec = _call("gpt-4o", "openai", req, resp, Usage(input_tokens=in_ct, output_tokens=out_ct))

    cand = Candidate(model="gpt-4o-mini", provider="openai")  # cheaper, but NO quality supplied
    report = evaluate_migration([rec], [cand])
    proj = report.ranked[0]

    assert proj.quality_confidence == QUALITY_UNKNOWN
    assert report.cost_only is True
    assert report.recommendation_safe is False
    # SEPARATE fields: EXACT cost (tiktoken) can coexist with unknown quality.
    if _HAVE_TIKTOKEN:
        assert proj.cost_confidence is Confidence.EXACT
    assert proj.quality_confidence == QUALITY_UNKNOWN
    # the honest note must be present
    assert any("quality unverified" in n for n in proj.notes)
    # NEVER name a switch on the unsafe path
    assert report.recommendation is None or "switch to" not in report.recommendation.lower()
    print("PASS test_no_quality_cost_only_no_switch")


# --- 5. with candidate quality -> cost_per_accepted, ranked by it, safe when real saving ---

def test_with_quality_cpa_ranked_and_safe():
    req = "summarise the following document into three crisp bullet points please"
    resp = "bullet one. bullet two. bullet three. that is the summary."
    in_ct, _ = count_tokens(req, "openai", "gpt-4o")
    out_ct, _ = count_tokens(resp, "openai", "gpt-4o")
    rec = _call("gpt-4o", "openai", req, resp, Usage(input_tokens=in_ct, output_tokens=out_ct))
    workload = [rec]

    # per_call_quality keyed by request_sha — accepted.
    per_call = {rec.request_sha: QualitySignal(status="accept")}
    cheaper = Candidate(model="gpt-4o-mini", provider="openai", per_call_quality=per_call)
    dearer = Candidate(model="o1", provider="openai", per_call_quality=per_call)  # pricier

    report = evaluate_migration(workload, [cheaper, dearer])
    assert report.cost_only is False                      # all candidates measured -> rank by CPA
    # ranked ascending by cost-per-accepted: gpt-4o-mini (cheap) first
    assert report.ranked[0].model == "gpt-4o-mini"
    assert isinstance(report.ranked[0].cost_per_accepted, float)
    # one accepted labeled call -> CPA == projected cost of that call
    assert abs(report.ranked[0].cost_per_accepted - report.ranked[0].projected_cost_usd) < 1e-12
    # real saving + measured quality -> safe, and may name the switch
    assert report.ranked[0].savings_usd > 0
    assert report.recommendation_safe is True
    assert "switch to gpt-4o-mini" in report.recommendation.lower()

    # accept_rate scalar path also yields a measured CPA.
    rate_cand = Candidate(model="gpt-4o-mini", provider="openai", accept_rate=0.5)
    rep2 = evaluate_migration(workload, [rate_cand])
    proj2 = rep2.ranked[0]
    assert proj2.quality_confidence == QUALITY_MEASURED
    # 1 call, rate 0.5 -> accepted 0.5 -> CPA = total cost / 0.5 = 2x cost
    assert abs(proj2.cost_per_accepted - proj2.projected_cost_usd / 0.5) < 1e-12
    print("PASS test_with_quality_cpa_ranked_and_safe")


def test_per_call_quality_wins_over_accept_rate():
    req = "a prompt"
    resp = "a response"
    rec = _call("gpt-4o", "openai", req, resp, Usage(input_tokens=50, output_tokens=40))
    # per_call accepted; accept_rate would imply 0.25. per_call must WIN -> CPA == single-call cost.
    cand = Candidate(model="gpt-4o-mini", provider="openai",
                     accept_rate=0.25,
                     per_call_quality={rec.request_sha: QualitySignal(status="accept")})
    rep = evaluate_migration([rec], [cand])
    proj = rep.ranked[0]
    assert abs(proj.cost_per_accepted - proj.projected_cost_usd) < 1e-12  # 1 accepted, not /0.25
    print("PASS test_per_call_quality_wins_over_accept_rate")


# --- 6. reasoning tokens reused as BOUNDED with the documented note ------------------------

def test_reasoning_reused_bounded_with_note():
    req = "a prompt that triggers reasoning"
    resp = "the visible answer"
    rec = _call("gpt-4o", "openai", req, resp,
                Usage(input_tokens=50, output_tokens=20, reasoning_tokens=300))
    cand = Candidate(model="gpt-4o-mini", provider="openai")
    report = evaluate_migration([rec], [cand])
    proj = report.ranked[0]

    # reasoning carried over unchanged (no text to re-tokenise)
    assert proj.projected_tokens.reasoning_tokens == 300
    # reasoning poisons cost confidence to BOUNDED even on a per_token candidate
    assert proj.cost_confidence is Confidence.BOUNDED
    assert any("reasoning effort" in n for n in proj.notes)
    print("PASS test_reasoning_reused_bounded_with_note")


# --- 7. no-egress: module source has no network import -----------------------------------

def test_no_network_import_in_module():
    import retoken.evaluator as ev
    src = open(ev.__file__, "r", encoding="utf-8").read()
    for bad in ("import requests", "import urllib", "import httpx",
                "import socket", "from urllib", "from requests", "from httpx",
                "import http.client", "socket.socket("):
        assert bad not in src, f"network import found: {bad!r}"
    print("PASS test_no_network_import_in_module")


# --- 8. purity: inputs unchanged after evaluate_migration --------------------------------

def test_inputs_unchanged_after_evaluate():
    req = "purity prompt"
    resp = "purity response"
    rec = _call("gpt-4o", "openai", req, resp,
                Usage(input_tokens=30, output_tokens=20, reasoning_tokens=5, cache_read_tokens=2))
    workload = [rec]
    cand = Candidate(model="gpt-4o-mini", provider="openai",
                     per_call_quality={rec.request_sha: QualitySignal(status="accept")})
    candidates = [cand]

    before_w = copy.deepcopy(workload)
    before_c = copy.deepcopy(candidates)
    evaluate_migration(workload, candidates)
    assert workload == before_w, "workload mutated"
    assert candidates == before_c, "candidates mutated"
    # the record's quality is still None (we never wrote candidate quality back onto it)
    assert workload[0].quality is None
    print("PASS test_inputs_unchanged_after_evaluate")


# --- serialisation sanity ----------------------------------------------------------------

def test_to_dict_serialises_report():
    rec = _call("gpt-4o", "openai", "p", "r", Usage(input_tokens=10, output_tokens=5))
    report = evaluate_migration([rec], [Candidate(model="gpt-4o-mini", provider="openai")])
    d = to_dict(report)
    assert isinstance(d, dict)
    assert isinstance(d["current_cost_usd"], float)
    # Confidence enum serialised to its string value
    assert d["ranked"][0]["cost_confidence"] in ("exact", "bounded")
    print("PASS test_to_dict_serialises_report")


def test_measured_quality_zero_accepted_never_asserts_switch():
    # REGRESSION (review panel catch): a candidate with quality SUPPLIED but ZERO accepted outputs
    # is NOT a measured quality delta. It must not crash and must NOT assert a switch, even when
    # cheaper. (Before the fix this hit the measured+savings gate, formatted an n/a string as
    # a float, and would have implied a safe switch.)
    rec = _call("gpt-4o", "openai", "prompt text here", "a response", Usage(input_tokens=40, output_tokens=30))
    per_call = {rec.request_sha: QualitySignal(status="reject")}  # supplied, but rejected
    cand = Candidate(model="gpt-4o-mini", provider="openai", per_call_quality=per_call)  # cheaper
    report = evaluate_migration([rec], [cand])  # must not raise
    top = report.ranked[0]
    assert top.quality_confidence == QUALITY_MEASURED      # evals WERE supplied
    assert not isinstance(top.cost_per_accepted, (int, float))  # but n/a — zero accepts
    assert report.recommendation_safe is False
    assert "switch to" not in (report.recommendation or "").lower()
    # honest WHY: it should say evals supplied / no accepted outputs, NOT "quality unverified"
    assert "no accepted outputs" in (report.recommendation or "").lower()
    print("PASS test_measured_quality_zero_accepted_never_asserts_switch")


def test_safe_recommendation_caveat_not_self_contradictory():
    # RUN-AND-OBSERVE catch: a safe (measured-quality) recommendation must NOT carry a
    # "quality unverified" caveat just because ANOTHER candidate lacks evals. The caveat must
    # describe the recommendation's basis. Realistic input: the cheapest candidate has evals,
    # a second candidate does not.
    rec = _call("gpt-4o", "openai", "summarise the report on revenue", "Revenue rose 12%.",
                Usage(input_tokens=120, output_tokens=80), quality=QualitySignal(status="accept"))
    cands = [
        Candidate(model="gpt-4o-mini", provider="openai", accept_rate=1.0),  # cheap + measured
        Candidate(model="o1", provider="openai"),                            # dearer + no evals
    ]
    report = evaluate_migration([rec], cands)
    assert report.recommendation_safe is True
    assert "switch to gpt-4o-mini" in (report.recommendation or "").lower()
    # the caveat must affirm the measured basis, NOT say "quality unverified"
    assert "quality unverified" not in report.caveat.lower()
    assert "measured quality" in report.caveat.lower()
    print("PASS test_safe_recommendation_caveat_not_self_contradictory")


def test_same_model_nets_zero_overcount_is_separate_reconciliation():
    # ADVISOR catch (cross-increment): current cost used REPORTED tokens while projection used
    # RE-TOKENISED tokens, so migrating a model to ITSELF showed the provider over-count as a fake
    # 'saving'. Now both sides use the independent basis -> same-model nets ~0, and the over-count
    # is surfaced as a SEPARATE reconciliation finding.
    req = "summarise the quarterly revenue report for the board in two sentences"
    resp = "Revenue rose twelve percent on EMEA demand and disciplined cost control"
    in_ct, _ = count_tokens(req, "openai", "gpt-4o")
    out_ct, _ = count_tokens(resp, "openai", "gpt-4o")
    # provider over-counts output by 40% (reported != re-tokenised)
    rec = _call("gpt-4o", "openai", req, resp,
                Usage(input_tokens=in_ct, output_tokens=int(out_ct * 1.4)))
    report = evaluate_migration([rec], [Candidate(model="gpt-4o", provider="openai")])
    p = report.ranked[0]
    # same model on the independent basis -> ~zero migration saving (NOT the 25% the bug showed)
    assert abs(p.savings_usd) < 1e-9, f"same-model migration must net ~0, got {p.savings_usd}"
    assert not report.recommendation_safe
    # the over-count is exposed SEPARATELY (reported bill > independent baseline)
    assert report.current_cost_reported_usd > report.current_cost_usd
    assert report.reconciliation_gap_usd > 0
    assert "reconciliation finding" in report.caveat.lower()
    # the honesty message must reach the RENDERED output, not just the object
    from retoken.evaluator import render_report
    assert "NOT a migration saving" in render_report(report)
    print("PASS test_same_model_nets_zero_overcount_is_separate_reconciliation")


def test_cache_tokens_reused_bounded_with_note():
    # Symmetric with the reasoning test: cache_read reused unchanged -> BOUNDED + documented note.
    rec = _call("gpt-4o", "openai", "prompt", "resp",
                Usage(input_tokens=30, output_tokens=20, cache_read_tokens=10))
    report = evaluate_migration([rec], [Candidate(model="gpt-4o-mini", provider="openai")])
    proj = report.ranked[0]
    assert proj.projected_tokens.cache_read_tokens == 10   # reused unchanged
    assert proj.cost_confidence is Confidence.BOUNDED
    assert any("cache behaviour" in n for n in proj.notes)
    print("PASS test_cache_tokens_reused_bounded_with_note")


def test_closed_candidate_cost_claim_is_bounded_not_verified():
    # HARNESS cross-cutting catch: a closed candidate (anthropic -> BOUNDED cost) with measured
    # quality + a saving may be recommendation_safe, but the caveat/recommendation must NOT claim
    # the cost is 'verified' — it rests on 'assumes equal tokenisation'. (Repro: opus -> sonnet.)
    req = "summarise the quarterly revenue report for the board in two sentences"
    resp = "Revenue rose twelve percent on EMEA demand and disciplined cost control"
    rec = _call("claude-opus-4", "anthropic", req, resp,
                Usage(input_tokens=200, output_tokens=150), quality=QualitySignal(status="accept"))
    cand = Candidate(model="claude-sonnet-4", provider="anthropic", accept_rate=0.9)  # cheaper, closed
    report = evaluate_migration([rec], [cand])
    top = report.ranked[0]
    assert top.cost_confidence is Confidence.BOUNDED          # closed candidate -> estimated cost
    assert report.recommendation_safe is True                 # quality measured + real saving
    # the cost claim must be honest about being an estimate, NEVER 'verified'
    assert "verified" not in report.caveat.lower()
    assert ("bounded" in report.caveat.lower() or "estimated" in report.caveat.lower())
    assert "estimated saving" in (report.recommendation or "").lower()
    print("PASS test_closed_candidate_cost_claim_is_bounded_not_verified")


def test_cli_candidate_id_with_colon_preserved():
    # RUN-AND-OBSERVE catch: a Bedrock model id ends in ':version'. The CLI must NOT split it into
    # provider/model on the colon (that turned 'amazon.nova-pro-v1:0' into model '0').
    from retoken.cli import _parse_candidate
    assert _parse_candidate("amazon.nova-pro-v1:0") == ("amazon", "amazon.nova-pro-v1:0")
    assert _parse_candidate("us.meta.llama3-1-70b-instruct-v1:0") == ("meta", "us.meta.llama3-1-70b-instruct-v1:0")
    assert _parse_candidate("anthropic.claude-3-5-sonnet-20240620-v1:0")[0] == "anthropic"
    assert _parse_candidate("gpt-4o-mini") == ("openai", "gpt-4o-mini")
    print("PASS test_cli_candidate_id_with_colon_preserved")


def test_render_report_shows_confidence_and_caveat():
    rec = _call("gpt-4o", "openai", "summarise the report", "Revenue rose twelve percent",
                Usage(input_tokens=50, output_tokens=40))
    from retoken.evaluator import render_report
    report = evaluate_migration([rec], [Candidate(model="gpt-4o-mini", provider="openai")])
    text = render_report(report)
    assert "Migration evaluation" in text
    assert "baseline (independent re-count)" in text
    assert "actual provider bill" in text
    assert "recommendation_safe=" in text
    assert "caveat:" in text
    # the confidence VALUES must actually render (would catch a deleted table column)
    assert ("exact" in text or "bounded" in text)        # cost confidence
    assert ("measured" in text or "unknown" in text)     # quality confidence
    print("PASS test_render_report_shows_confidence_and_caveat")


def test_cli_evaluate_end_to_end():
    # The headline feature must be runnable WITHOUT writing Python (advisor catch: was library-only).
    import json as _json
    from retoken.store import Store
    from retoken.cli import main
    fd, dbpath = tempfile.mkstemp(suffix=".db"); os.close(fd); os.remove(dbpath)
    jsonpath = dbpath.replace(".db", ".json")
    db = Store(dbpath)
    try:
        for i in range(4):
            db.record(_call("gpt-4o", "openai", "summarise the quarterly figures",
                            "Revenue rose twelve percent on strong demand", session_id=f"s{i}",
                            reported=Usage(input_tokens=200, output_tokens=150)))
        # cost-only (no quality) + a candidate with a supplied accept rate
        rc = main(["evaluate", "--db", dbpath, "--candidate", "gpt-4o-mini",
                   "--candidate", "o1", "--accept-rate", "gpt-4o-mini=0.9",
                   "--json", jsonpath])
        assert rc == 0
        with open(jsonpath, "r", encoding="utf-8") as f:
            d = _json.load(f)
        assert d["current_models"] == ["gpt-4o"]
        assert len(d["ranked"]) == 2
        # gpt-4o-mini has measured quality + a saving -> safe; o1 has none
        models = {r["model"] for r in d["ranked"]}
        assert models == {"gpt-4o-mini", "o1"}
        assert "cost_confidence" in d["ranked"][0]
        assert d["ranked"][0]["quality_confidence"] in ("measured", "unknown")
        # honesty assertions (would catch a regression in safe-gating or the caveat):
        # gpt-4o-mini has a supplied accept rate (measured) + a saving -> safe switch.
        assert d["recommendation_safe"] is True
        assert "switch to gpt-4o-mini" in (d["recommendation"] or "").lower()
        assert "measured quality" in d["caveat"]
    finally:
        for p in (dbpath, jsonpath):
            if os.path.exists(p):
                os.remove(p)
    print("PASS test_cli_evaluate_end_to_end")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all tests passed")
