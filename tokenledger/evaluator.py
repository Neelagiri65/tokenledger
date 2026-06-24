"""
Migration evaluator — project a captured workload onto CANDIDATE models, rank them, and
compute break-even across billing models. The migration headline's closer.

Honesty rule (load-bearing, from docs/plan-migration-and-cost-model.md A):
  "we prove the cheaper model's cost exactly; the quality-delta comes from YOUR evals, not ours."

So this module keeps TWO confidences strictly separate:
  - COST confidence (core.Confidence: EXACT / BOUNDED) — how well we know the dollars.
  - QUALITY confidence (a string: "measured" / "assumed" / "unknown") — whether a quality
    delta is backed by the customer's own eval/accept data. We NEVER assert a switch
    ("switch to X") without measured quality.

No data egress: re-tokenisation goes through core.count_tokens (local, offline). This module
imports no network library and opens no socket. It is PASSIVE — it never mutates the workload
records or the store.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from .core import (
    CallRecord, Usage, UsageAggregate, Confidence,
    CostModel, PerTokenCost, cost_model_for, count_tokens,
)
from .quality import QualitySignal


# Quality-confidence vocabulary. "assumed" is reserved for a future where we INFER a delta;
# for now this module only ever produces "measured" (customer supplied evals) or "unknown".
QUALITY_MEASURED = "measured"
QUALITY_ASSUMED = "assumed"   # reserved — not produced yet
QUALITY_UNKNOWN = "unknown"

_REASONING_NOTE = "assumes equivalent reasoning effort on the candidate"
_EQUAL_TOK_NOTE = "assumes equal tokenisation"
_CACHE_NOTE = "assumes equal cache behaviour (model/deployment specific)"
_NO_QUALITY_NOTE = ("cost delta only; quality unverified - supply candidate evals to confirm")


@dataclass
class Candidate:
    """A model we might migrate the workload to.

    cost_model overrides cost_model_for(model) when given (e.g. a partner's rented endpoint).
    accept_rate / per_call_quality are the CUSTOMER's measured quality data — without them we
    report cost only and never assert a switch. per_call_quality wins when both are present.
    """
    model: str
    provider: str = ""        # provider label for tokenizer routing; "" -> infer/blank
    cost_model: Optional[CostModel] = None
    accept_rate: Optional[float] = None                          # measured accept rate in [0,1]
    per_call_quality: Optional[dict[str, QualitySignal]] = None  # keyed by request_sha


@dataclass
class CandidateProjection:
    model: str
    projected_cost_usd: float
    cost_confidence: Confidence
    projected_tokens: UsageAggregate
    cost_per_accepted: Any            # float, or a string n/a marker
    quality_confidence: str          # "measured" / "assumed" / "unknown"
    savings_usd: float
    savings_pct: float
    breakeven_tokens: Optional[float]
    breakeven_note: str
    notes: list[str] = field(default_factory=list)


@dataclass
class MigrationReport:
    current_models: list[str]
    current_cost_usd: float           # migration BASELINE: independent (re-tokenised) basis
    ranked: list[CandidateProjection]
    cost_only: bool                  # True when ranking is by projected cost (quality unverified)
    recommendation: Optional[str]
    recommendation_safe: bool        # True ONLY if top candidate has measured quality AND a real saving
    caveat: str
    current_cost_reported_usd: float = 0.0  # the ACTUAL provider bill (reported tokens)
    reconciliation_gap_usd: float = 0.0     # reported - independent on the CURRENT model(s); a
                                            # provider count discrepancy, NOT a migration saving
    period_label: str = ""


# --- cost projection ------------------------------------------------------------------

_CPA_NA = "n/a (no accepted outputs yet)"


def _money(x: float) -> str:
    """Scale-aware dollars: cents for big figures, 4dp for sub-dollar so a small but real amount
    never rounds to a misleading $0.00 in prose."""
    return f"${x:,.2f}" if abs(x) >= 1 else f"${x:,.4f}"


def _candidate_cost_model(cand: Candidate) -> CostModel:
    return cand.cost_model if cand.cost_model is not None else cost_model_for(cand.model)


def _project_call_tokens(
    rec: CallRecord, cand: Candidate, notes_seen: set[str],
) -> tuple[Usage, bool]:
    """Re-cost ONE call's token buckets under the candidate. Returns the projected Usage and a
    flag `all_exact` = every CONTRIBUTING bucket is EXACT-sourced.

    Per the spec:
      - input/output: re-tokenise the captured text with the candidate's tokenizer. EXACT only
        if count_tokens returns EXACT; if no text OR BOUNDED -> REUSE the observed count,
        BOUNDED, note "assumes equal tokenisation".
      - reasoning: no text exists -> REUSE observed, BOUNDED, "assumes equivalent reasoning effort".
      - cache_read: reuse observed, BOUNDED (cache behaviour is model/deployment specific).

    A bucket with ZERO observed tokens is NOT contributing — it must not poison the confidence to
    BOUNDED. Otherwise a clean input/output-only workload could never be EXACT.
    """
    obs = rec.reported
    all_exact = True

    def _retokenise(text: str, observed: int) -> int:
        nonlocal all_exact
        if observed == 0:
            return 0  # not contributing — never poisons confidence, never re-tokenises empty
        if text:
            count, conf = count_tokens(text, cand.provider, cand.model)
            if conf is Confidence.EXACT:
                return count
            # BOUNDED tokenizer (closed candidate / no HF tokenizer offline) -> reuse observed.
            all_exact = False
            notes_seen.add(_EQUAL_TOK_NOTE)
            return observed
        # No captured text -> reuse observed, BOUNDED.
        all_exact = False
        notes_seen.add(_EQUAL_TOK_NOTE)
        return observed

    proj_input = _retokenise(rec.request_text, obs.input_tokens)
    proj_output = _retokenise(rec.response_text, obs.output_tokens)

    # reasoning: no text to re-tokenise -> reuse, BOUNDED (only if it contributes).
    proj_reasoning = obs.reasoning_tokens
    if obs.reasoning_tokens:
        all_exact = False
        notes_seen.add(_REASONING_NOTE)

    # cache_read: reuse, BOUNDED (only if it contributes).
    proj_cache = obs.cache_read_tokens
    if obs.cache_read_tokens:
        all_exact = False
        notes_seen.add(_CACHE_NOTE)

    return Usage(
        input_tokens=proj_input,
        output_tokens=proj_output,
        reasoning_tokens=proj_reasoning,
        cache_read_tokens=proj_cache,
    ), all_exact


def _project_cost(
    workload: list[CallRecord], cand: Candidate,
) -> tuple[float, Confidence, UsageAggregate, list[Usage], list[bool], list[str]]:
    """Re-cost the WHOLE workload under the candidate. Returns
    (projected_cost_usd, cost_confidence, projected_aggregate, per_call_projected_usages,
     per_call_all_exact, notes).

    Capacity models (flat/rented) are amortised as effective $/token over the PROJECTED period
    (sum of projected tokens) then x each call's projected tokens — mirroring
    dashboard._capacity_effective_rates so the per-call shares sum to the period figure exactly.
    """
    cm = _candidate_cost_model(cand)
    notes_seen: set[str] = set()

    proj_usages: list[Usage] = []
    all_exact_flags: list[bool] = []
    agg = UsageAggregate()
    for rec in workload:
        pu, all_exact = _project_call_tokens(rec, cand, notes_seen)
        proj_usages.append(pu)
        all_exact_flags.append(all_exact)
        agg.add_usage(pu)

    every_bucket_exact = all(all_exact_flags) if all_exact_flags else True

    if isinstance(cm, PerTokenCost):
        # Linear: period cost == sum of per-call costs.
        total = sum(cm.call_cost(u).usd for u in proj_usages)
        # 3e: EXACT only if every contributing token count is EXACT-sourced AND per_token.
        cost_conf = Confidence.EXACT if every_bucket_exact else Confidence.BOUNDED
    else:
        # Capacity model: effective $/token over the projected period, x each call's tokens.
        period = cm.period_cost(agg)
        eff = period.effective_per_token or 0.0
        total = 0.0
        for u in proj_usages:
            call_tokens = (u.input_tokens + u.output_tokens
                           + u.reasoning_tokens + u.cache_read_tokens)
            total += eff * call_tokens
        cost_conf = Confidence.BOUNDED  # capacity cost is always BOUNDED
        if period.note:
            notes_seen.add(period.note)

    return total, cost_conf, agg, proj_usages, all_exact_flags, sorted(notes_seen)


# --- current cost ---------------------------------------------------------------------

def _reported_cost(workload: list[CallRecord]) -> float:
    """The ACTUAL provider bill: provider-REPORTED tokens x the current model's rate. This is what
    the customer is billed today (incl. any over-count). Kept SEPARATE from the migration baseline
    so a provider count discrepancy is never smuggled into 'savings'."""
    cap_aggs: dict[str, UsageAggregate] = {}
    for rec in workload:
        if not isinstance(cost_model_for(rec.model), PerTokenCost):
            cap_aggs.setdefault(rec.model, UsageAggregate()).add_usage(rec.reported)
    cap_eff = {m: cost_model_for(m).period_cost(agg).effective_per_token or 0.0
               for m, agg in cap_aggs.items()}
    total = 0.0
    for rec in workload:
        cm = cost_model_for(rec.model)
        if isinstance(cm, PerTokenCost):
            total += cm.call_cost(rec.reported).usd
        else:
            u = rec.reported
            total += cap_eff.get(rec.model, 0.0) * (
                u.input_tokens + u.output_tokens + u.reasoning_tokens + u.cache_read_tokens)
    return total


def _current_cost(workload: list[CallRecord]) -> tuple[float, list[str]]:
    """The migration BASELINE: the current model(s) costed on the SAME INDEPENDENT (re-tokenised)
    basis used for candidates — each call re-counted under its OWN current model. This isolates the
    model/rate decision: migrating a model to ITSELF nets ~zero, instead of showing the provider's
    over-count as a fake 'saving'. The over-count is a SEPARATE reconciliation finding (see
    _reported_cost / the dashboard). Mirrors _project_cost so both sides of a comparison are
    apples-to-apples."""
    by_model: dict[str, list[CallRecord]] = {}
    models: list[str] = []
    for rec in workload:
        if rec.model not in by_model:
            by_model[rec.model] = []
            models.append(rec.model)
        by_model[rec.model].append(rec)
    total = 0.0
    for model, recs in by_model.items():
        # treat the current model as its own candidate -> re-tokenise under it (independent basis).
        cand = Candidate(model=model, provider=recs[0].provider)
        cost, _conf, _agg, _pu, _flags, _notes = _project_cost(recs, cand)
        total += cost
    return total, models


# --- quality (cost-per-accepted) ------------------------------------------------------

def _candidate_cost_per_accepted(
    workload: list[CallRecord], cand: Candidate,
    proj_usages: list[Usage],
) -> tuple[Any, str]:
    """Candidate cost-per-accepted, reusing dashboard.cost_per_accepted semantics:
      numerator = projected cost over LABELED calls (incl. rejected calls' cost — the FinOps
                  waste signal), denominator = accepted count. accept is the SOLE driver;
                  unknown/None is NEVER a reject (it is unlabeled, excluded).

    Quality source precedence (documented): per_call_quality WINS when both present.
      - per_call_quality: per-call labels keyed by request_sha.
      - accept_rate (scalar): no per-call labels exist, so we read EVERY call as labeled and the
        accepted count as accept_rate x N_calls. (Documented denominator choice for the scalar
        case — the simplest coherent reading the spec leaves open.)

    Returns (cost_per_accepted_value_or_na_string, quality_confidence).
    """
    cm = _candidate_cost_model(cand)
    has_per_call = bool(cand.per_call_quality)
    has_rate = cand.accept_rate is not None
    if not has_per_call and not has_rate:
        return _CPA_NA, QUALITY_UNKNOWN

    # Per-call projected cost (for the labeled-call numerator). Per-token: exact per-call.
    # Capacity: effective $/token over the projected period x each call's projected tokens.
    if isinstance(cm, PerTokenCost):
        per_call_cost = [cm.call_cost(u).usd for u in proj_usages]
    else:
        agg = UsageAggregate()
        for u in proj_usages:
            agg.add_usage(u)
        eff = cm.period_cost(agg).effective_per_token or 0.0
        per_call_cost = [
            eff * (u.input_tokens + u.output_tokens + u.reasoning_tokens + u.cache_read_tokens)
            for u in proj_usages
        ]

    if has_per_call:
        labeled_cost = 0.0
        accepted = 0
        for rec, cost in zip(workload, per_call_cost):
            q = cand.per_call_quality.get(rec.request_sha)
            if q is not None and q.is_labeled():
                labeled_cost += cost            # numerator includes rejected calls' cost
                if q.is_accepted():
                    accepted += 1
        if accepted <= 0:
            return _CPA_NA, QUALITY_MEASURED
        return labeled_cost / accepted, QUALITY_MEASURED

    # accept_rate scalar path: every call is labeled; accepted = rate x N.
    rate = max(0.0, min(1.0, float(cand.accept_rate)))
    n = len(workload)
    accepted = rate * n
    if accepted <= 0:
        return _CPA_NA, QUALITY_MEASURED
    total_labeled_cost = sum(per_call_cost)     # all calls labeled in the scalar reading
    return total_labeled_cost / accepted, QUALITY_MEASURED


# --- break-even -----------------------------------------------------------------------

@dataclass
class BreakEven:
    tokens: Optional[float]
    usd_at_breakeven: Optional[float]
    note: str


def breakeven(cost_model: CostModel, baseline_rate_per_token: float) -> BreakEven:
    """Monthly token volume V where a CAPACITY candidate's PERIOD cost equals V x baseline rate.

    baseline_rate_per_token is in $/TOKEN (NOT $/1M). e.g. $2/1M tokens -> 2.0/1e6.
    - flat: period cost = fee -> V = fee / b.
    - rented (provisioned gpu_hours): period cost is fixed -> V = period_cost / b.
    - rented (throughput-only, no fixed period cost) OR per_token candidate: cost is LINEAR in
      volume -> no break-even -> None.

    The capacity period cost is read uniformly from period_cost(UsageAggregate()) — with zero
    measured tokens it returns the FIXED period figure (fee, or provisioned GPU cost) for flat /
    rented-with-hours, and 0/None for the linear shapes.
    """
    if baseline_rate_per_token <= 0:
        return BreakEven(None, None, "baseline rate must be > 0 $/token")
    if isinstance(cost_model, PerTokenCost):
        return BreakEven(None, None,
                         "per_token vs per_token is linear — no break-even volume")
    # Capacity: the fixed period figure with no measured tokens.
    fixed = cost_model.period_cost(UsageAggregate()).usd
    if not fixed:
        # rented throughput-only has no fixed period cost -> linear, no break-even.
        return BreakEven(None, None,
                         "candidate has no fixed period cost (linear) — no break-even volume")
    v = fixed / baseline_rate_per_token
    return BreakEven(v, fixed,
                     f"period cost ${fixed:,.2f} == {v:,.0f} tokens x "
                     f"${baseline_rate_per_token * 1e6:,.4f}/1M baseline")


def _baseline_rate_per_token(workload: list[CallRecord]) -> Optional[float]:
    """A per_token baseline $/token from the workload's current per-token model(s), for the
    break-even comparison. Uses the blended effective $/token over reported usage of the per-token
    portion. Returns None if the workload has no per-token cost (e.g. all capacity)."""
    agg = UsageAggregate()
    usd = 0.0
    for rec in workload:
        cm = cost_model_for(rec.model)
        if isinstance(cm, PerTokenCost):
            usd += cm.call_cost(rec.reported).usd
            agg.add_usage(rec.reported)
    toks = agg.total_tokens
    if toks <= 0 or usd <= 0:
        return None
    return usd / toks


# --- top-level ------------------------------------------------------------------------

def evaluate_migration(
    workload: list[CallRecord],
    candidates: list[Candidate],
    period_label: str = "",
) -> MigrationReport:
    """Project a captured workload onto each candidate, rank, and build a MigrationReport.

    PASSIVE: does not mutate the workload records or any store. Cost and quality confidence are
    labelled SEPARATELY. A switch is NEVER asserted without measured quality.
    """
    current_cost, current_models = _current_cost(workload)   # independent (re-tokenised) baseline
    reported_cost = _reported_cost(workload)                  # the actual provider bill
    reconciliation_gap = reported_cost - current_cost        # over/under-count $, a SEPARATE finding
    baseline_rate = _baseline_rate_per_token(workload)

    projections: list[CandidateProjection] = []
    for cand in candidates:
        cost, cost_conf, agg, proj_usages, _flags, notes = _project_cost(workload, cand)
        cpa, q_conf = _candidate_cost_per_accepted(workload, cand, proj_usages)

        savings = current_cost - cost
        savings_pct = (savings / current_cost * 100.0) if current_cost else 0.0

        cm = _candidate_cost_model(cand)
        if baseline_rate is not None:
            be = breakeven(cm, baseline_rate)
        else:
            be = BreakEven(None, None, "no per-token baseline in the workload — break-even N/A")

        if q_conf == QUALITY_UNKNOWN:
            notes = list(notes) + [_NO_QUALITY_NOTE]

        projections.append(CandidateProjection(
            model=cand.model,
            projected_cost_usd=cost,
            cost_confidence=cost_conf,
            projected_tokens=agg,
            cost_per_accepted=cpa,
            quality_confidence=q_conf,
            savings_usd=savings,
            savings_pct=savings_pct,
            breakeven_tokens=be.tokens,
            breakeven_note=be.note,
            notes=notes,
        ))

    # Ranking: by cost-per-accepted ONLY if every candidate has measured quality with a usable
    # numeric CPA; otherwise by projected cost ascending (COST-ONLY ranking flag set).
    all_measured = bool(projections) and all(
        p.quality_confidence == QUALITY_MEASURED and isinstance(p.cost_per_accepted, (int, float))
        for p in projections
    )
    if all_measured:
        ranked = sorted(projections, key=lambda p: p.cost_per_accepted)
        cost_only = False
    else:
        ranked = sorted(projections, key=lambda p: p.projected_cost_usd)
        cost_only = True

    # Recommendation. recommendation_safe is True ONLY when the top candidate has measured quality
    # AND a real cost saving. Never name a "switch" on the unsafe path.
    recommendation: Optional[str] = None
    recommendation_safe = False
    if ranked:
        top = ranked[0]
        # SAFE only with measured quality, a NUMERIC cost-per-accepted (measured quality with ZERO
        # accepted outputs is NOT a measured quality delta — it must not assert a switch; this guard
        # mirrors the line-401 ranking gate), and a real saving.
        if (top.quality_confidence == QUALITY_MEASURED
                and isinstance(top.cost_per_accepted, (int, float))
                and top.savings_usd > 0):
            recommendation_safe = True
            # The cost claim MUST reflect cost_confidence: for a closed candidate (Anthropic/Gemini)
            # the projected cost is BOUNDED (assumes equal tokenisation), so we must NOT imply the
            # dollar saving is exact. (Harness cross-cutting review caught the over-claim.)
            _exact = top.cost_confidence is Confidence.EXACT
            cost_phrase = ("cost re-counted exactly" if _exact
                           else "cost is BOUNDED — assumes equal tokenisation for this closed model")
            saving_word = "saving" if _exact else "estimated saving"
            recommendation = (
                f"switch to {top.model}: projected {saving_word} {_money(top.savings_usd)} "
                f"({top.savings_pct:.1f}%) at measured quality "
                f"(cost-per-accepted {top.cost_per_accepted:.6f}); {cost_phrase}"
            )
        else:
            # Unsafe: cost-only insight, never an asserted switch. Be honest about WHY it is unsafe:
            # quality genuinely unknown (no evals) vs. evals supplied but no accepted outputs to judge.
            if top.quality_confidence == QUALITY_MEASURED:
                why = ("quality evals supplied but no accepted outputs to judge on; "
                       "cannot assert a switch")
            else:
                why = _NO_QUALITY_NOTE
            recommendation = (
                f"cheapest candidate by cost is {top.model} "
                f"(projected {_money(top.projected_cost_usd)}); {why}"
            )

    # Caveat must describe the RECOMMENDATION's basis, not contradict it. When the top candidate is
    # a SAFE switch (measured quality + saving), saying "quality unverified" would be false even if
    # OTHER candidates lack evals — so the recommendation-safe case is handled first, and only then
    # do we note that the rest of the field is ranked on cost. (Run-and-observe caught a report that
    # was simultaneously recommendation_safe AND carried a "quality unverified" caveat.)
    any_unknown_quality = any(p.quality_confidence == QUALITY_UNKNOWN for p in projections)
    if recommendation_safe:
        _exact = ranked[0].cost_confidence is Confidence.EXACT
        cost_phrase = ("cost is verified locally (exact re-count)" if _exact
                       else "cost is ESTIMATED locally (BOUNDED — assumes equal tokenisation for "
                            "this closed model; the dollar saving is approximate)")
        caveat = ("recommended switch is backed by customer-supplied evals (measured quality); "
                  + cost_phrase)
        if any_unknown_quality:
            caveat += ". Other candidates lack quality data and are listed by cost only"
    elif any_unknown_quality:
        caveat = _NO_QUALITY_NOTE
    elif cost_only:
        caveat = ("quality supplied for all candidates but no accepted outputs to rank on; "
                  "ranked by projected cost — cost is verified locally")
    else:
        caveat = ("ranking by cost-per-accepted using customer-supplied evals; "
                  "cost is verified locally, quality delta comes from your evals")

    # A material reported-vs-independent gap on the CURRENT model is a reconciliation finding, NOT a
    # migration saving — surface it separately so it is never conflated with the switch decision.
    if current_cost > 0 and abs(reconciliation_gap) > 0.005 * current_cost:
        sign = "over" if reconciliation_gap > 0 else "under"
        caveat += (f". Separately: the current provider {sign}-counts by "
                   f"{_money(abs(reconciliation_gap))} vs our independent re-count "
                   f"(reconciliation finding, not a migration saving)")

    return MigrationReport(
        current_models=current_models,
        current_cost_usd=current_cost,
        ranked=ranked,
        cost_only=cost_only,
        recommendation=recommendation,
        recommendation_safe=recommendation_safe,
        caveat=caveat,
        current_cost_reported_usd=reported_cost,
        reconciliation_gap_usd=reconciliation_gap,
        period_label=period_label,
    )


def render_report(report: MigrationReport) -> str:
    """Human-readable text rendering of a MigrationReport (for the CLI). Cost confidence and
    quality confidence are shown SEPARATELY; the recommendation safety and caveat are printed
    verbatim so the honesty labels reach the reader."""
    L: list[str] = []
    title = "Migration evaluation"
    if report.period_label:
        title += f" — {report.period_label}"
    L.append(title)
    L.append(f"current model(s): {', '.join(report.current_models) or 'n/a'}")
    L.append(f"baseline (independent re-count): {_money(report.current_cost_usd)}    "
             f"actual provider bill (reported): {_money(report.current_cost_reported_usd)}")
    if report.current_cost_usd > 0 and abs(report.reconciliation_gap_usd) > 0.005 * report.current_cost_usd:
        sign = "over" if report.reconciliation_gap_usd > 0 else "under"
        L.append(f"  reconciliation: provider {sign}-counts {_money(abs(report.reconciliation_gap_usd))} "
                 f"vs our re-count (separate finding, NOT a migration saving)")
    L.append("")
    L.append(f"{'candidate':<22}{'proj $':>12}{'cost':>9}{'save $':>12}{'save %':>8}"
             f"{'$/accepted':>14}{'quality':>10}")
    L.append("-" * 87)
    for p in report.ranked:
        cpa = f"${p.cost_per_accepted:.4f}" if isinstance(p.cost_per_accepted, (int, float)) else "n/a"
        L.append(f"{p.model:<22}{p.projected_cost_usd:>12.4f}{p.cost_confidence.value:>9}"
                 f"{p.savings_usd:>12.4f}{p.savings_pct:>7.1f}%{cpa:>14}{p.quality_confidence:>10}")
    L.append("")
    L.append(f"recommendation: {report.recommendation or 'n/a'}")
    L.append(f"  (recommendation_safe={report.recommendation_safe})")
    L.append(f"caveat: {report.caveat}")
    return "\n".join(L)


def to_dict(obj: Any) -> dict:
    """Serialise a dataclass (MigrationReport / CandidateProjection / ...) consistent with
    core.to_dict. Enums become their values; the n/a CPA string passes through."""
    return _serialise(asdict(obj))


def _serialise(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _serialise(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialise(v) for v in value]
    if isinstance(value, Confidence):
        return value.value
    return value
