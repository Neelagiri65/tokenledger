"""
Dashboard + reporting. Produces three views from the stored calls:
  - a CLI summary (console),
  - a self-contained HTML dashboard (no server, no JS deps),
  - a markdown "discrepancy report" you can take to a provider to discuss billing.
All computed locally from the SQLite store.
"""

from __future__ import annotations

import html
from collections import defaultdict
from dataclasses import dataclass, field

from .core import (
    CallReconciliation, Verdict, PerTokenCost, UsageAggregate, Usage, _cost,
    cost_model_for, reconcile_call,
)
from .store import Store


def _independent_call_usd(rc: CallReconciliation) -> float:
    """Cost from OUR re-counted tokens (the reconcile buckets' independent counts), NOT the provider's
    reported tokens — the real dollar-level verification. Output uses the re-count (exact where a
    tokenizer exists); input uses the bounded re-count; reasoning/cache are reused (unverifiable)."""
    r = rc.record
    bk = {b.bucket: b for b in rc.buckets}
    out_b, in_b = bk.get("output"), bk.get("input")
    out_tok = out_b.independent if (out_b and out_b.independent is not None) else r.reported.output_tokens
    in_total = in_b.independent if (in_b and in_b.independent is not None) else (
        r.reported.input_tokens + r.reported.cache_read_tokens)
    cache = r.reported.cache_read_tokens
    u = Usage(input_tokens=max(0, in_total - cache), output_tokens=out_tok,
              reasoning_tokens=r.reported.reasoning_tokens, cache_read_tokens=cache,
              cache_creation_tokens=r.reported.cache_creation_tokens)
    cm = cost_model_for(r.model)
    return cm.call_cost(u).usd if isinstance(cm, PerTokenCost) else 0.0


@dataclass
class Rollup:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    billed_usd: float = 0.0
    flagged: int = 0
    overbilled_output_tokens: int = 0   # exact-check surplus only
    overbilled_usd: float = 0.0
    # Quality coverage (cost-per-accepted). accept is the SOLE accepted-count driver; unknown/None
    # is NEVER a reject — it is unlabeled and excluded from the metric.
    labeled_calls: int = 0              # status in {accept, reject}
    accepted_calls: int = 0            # status == accept
    unlabeled_calls: int = 0          # status None or 'unknown'
    labeled_billed_usd: float = 0.0   # billed over labeled calls = cost-per-accepted NUMERATOR
    # Cost confidence surface. A capacity (flat/rented) model contributes a BOUNDED effective-$/token
    # cost, so any bucket it touches has a partly-bounded $ figure — marked here so the views never
    # show a capacity cost as if it were an exact per-token bill. cost_notes carries the per-model
    # period notes (e.g. rented utilisation %) for display.
    cost_bounded: bool = False        # True if any capacity model contributed to billed_usd
    cost_notes: list = field(default_factory=list)
    # The provider's OWN reported $ (read from logs, e.g. LiteLLM `spend`) — the ACTUAL charge, no
    # typed rates. Summed only over calls that carried it; reported_cost_calls says how many.
    provider_reported_usd: float = 0.0
    reported_cost_calls: int = 0
    # cost from OUR INDEPENDENT re-count (not the provider's reported tokens) — the real verification.
    # BOUNDED overall (output re-tokenised exactly where a tokenizer exists; input bounded; reasoning/
    # cache reused as unverifiable). billed_usd above uses REPORTED tokens; this uses re-counted ones.
    independent_usd: float = 0.0


_CPA_NA = "n/a (no accepted outputs yet)"
_CPA_CAVEAT = ("Cost-per-accepted is DESCRIPTIVE. The thesis that it should drive model switching "
               "is UNVALIDATED; this is not a migration recommendation.")


def cost_per_accepted(ru: Rollup):
    """(sum billed_usd over calls with a KNOWN accept/reject label) / (count of status=='accept').

    Charges the cost of rejected calls against the accepted ones — the FinOps waste signal.
    Unlabeled (None/'unknown') calls are excluded from both. When the denominator is 0 (no accepts,
    or no labeled calls at all) returns the literal n/a string — never divides, 0, or raises."""
    if ru.accepted_calls <= 0:
        return _CPA_NA
    return ru.labeled_billed_usd / ru.accepted_calls


def reconcile_all(store: Store, num_messages: int = 1) -> list[CallReconciliation]:
    return [reconcile_call(r, num_messages=num_messages) for r in store.all_records()]


def _capacity_results(recs: list[CallReconciliation]) -> dict[str, "CostResult"]:
    """For each non-per-token (flat/rented) model, its period CostResult over the WHOLE period
    (all recs) — carrying the effective $/token AND the period note (e.g. rented utilisation %).
    A period fee must be amortised per-call as eff x call_tokens against the GLOBAL total — not
    re-applied per rollup bucket — so the cost sums to the period figure exactly under any
    partition (provider / activity / session) and cost-per-accepted is the labeled token share."""
    aggs: dict[str, UsageAggregate] = defaultdict(UsageAggregate)
    for rc in recs:
        m = rc.record.model
        if not isinstance(cost_model_for(m), PerTokenCost):
            aggs[m].add_usage(rc.record.reported)
    return {m: cost_model_for(m).period_cost(agg) for m, agg in aggs.items()}


def rollup_by(recs: list[CallReconciliation], key: str) -> dict[str, Rollup]:
    out: dict[str, Rollup] = defaultdict(Rollup)
    cap_res = _capacity_results(recs)  # period CostResult per capacity model
    cap_eff = {m: r.effective_per_token for m, r in cap_res.items()
               if r.effective_per_token is not None}
    for rc in recs:
        r = rc.record
        k = getattr(r, key)
        ru = out[k]
        ru.calls += 1
        ru.input_tokens += r.reported.input_tokens
        ru.output_tokens += r.reported.output_tokens
        ru.reasoning_tokens += r.reported.reasoning_tokens
        ru.cache_read_tokens += r.reported.cache_read_tokens
        cm = cost_model_for(r.model)
        if isinstance(cm, PerTokenCost):
            billed = cm.call_cost(r.reported).usd     # EXACT per-call (== legacy _cost)
        else:
            # Capacity model: amortise the period fee as eff $/token x this call's tokens. Summed
            # across all calls this recovers the period figure exactly (eff x Σtokens = period_cost).
            u = r.reported
            call_tokens = u.input_tokens + u.output_tokens + u.reasoning_tokens + u.cache_read_tokens
            billed = cap_eff.get(r.model, 0.0) * call_tokens
            ru.cost_bounded = True   # capacity cost is BOUNDED, not an exact per-token bill
            res = cap_res.get(r.model)
            if res is not None and res.note and res.note not in ru.cost_notes:
                ru.cost_notes.append(res.note)
        ru.billed_usd += billed
        ru.independent_usd += _independent_call_usd(rc)   # cost from OUR re-count (the real check)
        if r.reported_cost_usd is not None:           # the provider's actual charge, when logged
            ru.provider_reported_usd += r.reported_cost_usd
            ru.reported_cost_calls += 1
        # Quality coverage. accept is the SOLE accepted driver; eval_score/success are NOT OR-ed in.
        q = r.quality
        if q is not None and q.is_labeled():
            ru.labeled_calls += 1
            ru.labeled_billed_usd += billed           # numerator includes rejected calls' cost
            if q.is_accepted():
                ru.accepted_calls += 1
        else:
            ru.unlabeled_calls += 1                    # None or 'unknown' — never read as a reject
        if rc.has_overcount:
            ru.flagged += 1
        for b in rc.buckets:
            # EXACT overcount ONLY gets a precise dollar dispute. OUT_OF_BAND is the estimate-based
            # (closed-provider, BOUNDED) path where b.independent is a crude word-count estimate, not
            # a re-tokenised count — charging reported-minus-estimate into the precise 'overbill $'
            # column presents estimator wobble as an exact overbilling (a false-precision/trust bug
            # the harness cross-cutting review caught). OUT_OF_BAND still flags the call (has_overcount)
            # and appears in the discrepancy report as an estimate-band finding — just not as $.
            if b.bucket == "output" and b.verdict == Verdict.OVERCOUNT and b.independent is not None:
                surplus = max(0, b.reported - b.independent)
                ru.overbilled_output_tokens += surplus
                price = _price(r.model)
                if price is not None:                  # per-token billable: dispute the surplus in $
                    ru.overbilled_usd += surplus / 1e6 * price[1]
                # capacity model: token surplus tracked, but there is no per-token bill to put a
                # dollar figure on — overbilled_usd intentionally left untouched.
    return out


def _price(model: str):
    """Per-token (input, output, reasoning, cache_read) rates for the model, or None when the
    model is NOT per-token billable (flat/rented) — there is no per-token bill to reconcile a
    dollar discrepancy against."""
    cm = cost_model_for(model)
    if isinstance(cm, PerTokenCost):
        return (cm.input, cm.output, cm.reasoning, cm.cache_read)
    return None


_BOUNDED_LEGEND = ("~ marks a cost that INCLUDES a capacity model (flat/rented): a BOUNDED effective "
                   "$/token amortised over measured usage, not an exact per-token bill.")


def _billed_cell(ru: Rollup) -> str:
    """Billed $ with a '~' marker when the figure includes BOUNDED capacity-model cost."""
    return ("~%.4f" if ru.cost_bounded else "%.4f") % ru.billed_usd


def print_summary(store: Store, num_messages: int = 1) -> None:
    recs = reconcile_all(store, num_messages)
    print(f"\nTokenLedger — {len(recs)} calls across "
          f"{len({r.record.provider for r in recs})} providers, "
          f"{len({r.record.session_id for r in recs})} sessions\n")

    by_prov = rollup_by(recs, "provider")
    print(f"{'provider':<14}{'calls':>7}{'in':>12}{'out':>12}{'reason':>10}{'billed $':>12}{'flagged':>9}{'overbill $':>12}")
    print("-" * 88)
    tot = Rollup()
    for prov, ru in sorted(by_prov.items()):
        print(f"{prov:<14}{ru.calls:>7}{ru.input_tokens:>12,}{ru.output_tokens:>12,}"
              f"{ru.reasoning_tokens:>10,}{_billed_cell(ru):>12}{ru.flagged:>9}{ru.overbilled_usd:>12.4f}")
        tot.calls += ru.calls; tot.input_tokens += ru.input_tokens
        tot.output_tokens += ru.output_tokens; tot.reasoning_tokens += ru.reasoning_tokens
        tot.billed_usd += ru.billed_usd; tot.flagged += ru.flagged
        tot.overbilled_usd += ru.overbilled_usd
        tot.provider_reported_usd += ru.provider_reported_usd
        tot.reported_cost_calls += ru.reported_cost_calls
        tot.independent_usd += ru.independent_usd
        if ru.cost_bounded:
            tot.cost_bounded = True
        for n in ru.cost_notes:
            if n not in tot.cost_notes:
                tot.cost_notes.append(n)
    print("-" * 88)
    print(f"{'TOTAL':<14}{tot.calls:>7}{tot.input_tokens:>12,}{tot.output_tokens:>12,}"
          f"{tot.reasoning_tokens:>10,}{_billed_cell(tot):>12}{tot.flagged:>9}{tot.overbilled_usd:>12.4f}")
    if tot.cost_bounded:
        print(f"\n{_BOUNDED_LEGEND}")
        for n in tot.cost_notes:
            print(f"  · {n}")
    # The verification line, three HONEST numbers:
    #  billed (provider's reported tokens x rate) vs independent re-count (OUR re-counted tokens x rate)
    #  — the independent figure is the actual check; it is BOUNDED (output re-tokenised exactly where a
    #  tokenizer exists, input bounded, reasoning/cache unverifiable). Provider-reported $ (their actual
    #  charge from the logs) shown when available.
    delta = tot.billed_usd - tot.independent_usd
    print(f"\nBilled (provider's reported tokens): ${tot.billed_usd:.4f}    "
          f"Independent re-count (ours, BOUNDED): ${tot.independent_usd:.4f}    "
          f"Δ ${delta:+.4f}")
    print("  The independent figure is computed from OUR re-count of the actual text, not the "
          "provider's token counts — output re-tokenised exactly where a tokenizer exists, input "
          "bounded, reasoning/cache unverifiable. Per-call over-counts are in the 'flagged'/'overbill' columns.")
    if tot.reported_cost_calls:
        print(f"  Provider-reported $ from logs ({tot.reported_cost_calls}/{tot.calls} calls): "
              f"${tot.provider_reported_usd:.4f} (their actual charge).")

    # By activity type — what the tokens were actually spent ON.
    by_act = rollup_by(recs, "task_class")
    print(f"\n{'activity':<16}{'calls':>7}{'in':>12}{'out':>12}{'reason':>10}{'billed $':>12}")
    print("-" * 69)
    for act, ru in sorted(by_act.items(), key=lambda kv: -kv[1].billed_usd):
        print(f"{act:<16}{ru.calls:>7}{ru.input_tokens:>12,}{ru.output_tokens:>12,}"
              f"{ru.reasoning_tokens:>10,}{_billed_cell(ru):>12}")

    # Cost per accepted output — what the spend actually BOUGHT, by activity.
    print(f"\n{'activity':<16}{'accepted':>9}{'labeled':>9}{'unlabeled':>11}"
          f"{'labeled $':>12}{'$/accepted':>32}")
    print("-" * 89)
    tot_labeled = tot_unlabeled = 0
    for act, ru in sorted(by_act.items(), key=lambda kv: -kv[1].billed_usd):
        cpa = cost_per_accepted(ru)
        cpa_s = cpa if isinstance(cpa, str) else f"${cpa:.4f}"
        print(f"{act:<16}{ru.accepted_calls:>9}{ru.labeled_calls:>9}{ru.unlabeled_calls:>11}"
              f"{ru.labeled_billed_usd:>12.4f}{cpa_s:>32}")
        tot_labeled += ru.labeled_calls; tot_unlabeled += ru.unlabeled_calls
    print(f"Cost-per-accepted computed over {tot_labeled} labeled call(s); {tot_unlabeled} unlabeled.")
    print(_CPA_CAVEAT)

    flagged = [rc for rc in recs if rc.has_overcount]
    if flagged:
        print(f"\n{len(flagged)} call(s) with a billing discrepancy:")
        for rc in flagged:
            for b in rc.buckets:
                if b.verdict in (Verdict.OVERCOUNT, Verdict.OUT_OF_BAND):
                    print(f"  [{rc.record.provider}/{rc.record.model}] session={rc.record.session_id} "
                          f"{b.bucket}: {b.verdict.value} — {b.note}")
    print()


def discrepancy_report_md(store: Store, num_messages: int = 1) -> str:
    recs = reconcile_all(store, num_messages)
    flagged = [rc for rc in recs if rc.has_overcount]
    lines = ["# Token billing discrepancy report", ""]
    lines.append(f"Calls audited: {len(recs)}. Discrepancies: {len(flagged)}.")
    lines.append("")
    lines.append("Method: output tokens re-tokenized from the returned text with the "
                 "provider's own tokenizer (exact where available); input bounded against "
                 "the sent payload plus documented overhead. Reasoning and cache buckets "
                 "are recorded but not disputed here (not verifiable per call).")
    lines.append("")
    if not flagged:
        lines.append("No output-token over-counts or out-of-band figures detected.")
        return "\n".join(lines)
    lines.append("| provider | model | session | bucket | billed | independent | finding |")
    lines.append("|---|---|---|---|---|---|---|")
    for rc in flagged:
        for b in rc.buckets:
            if b.verdict in (Verdict.OVERCOUNT, Verdict.OUT_OF_BAND):
                ind = b.independent if b.independent is not None else "n/a"
                lines.append(f"| {rc.record.provider} | {rc.record.model} | {rc.record.session_id} "
                             f"| {b.bucket} | {b.reported} | {ind} | {b.verdict.value}: {b.note} |")
    return "\n".join(lines)


def write_html(store: Store, path: str = "tokenledger.html", num_messages: int = 1) -> str:
    recs = reconcile_all(store, num_messages)
    by_prov = rollup_by(recs, "provider")
    by_sess = rollup_by(recs, "session_id")
    by_act = rollup_by(recs, "task_class")

    def rows(d: dict[str, Rollup], label: str) -> str:
        out = []
        for k, ru in sorted(d.items()):
            cls = "flag" if ru.flagged else ""
            # BOUNDED marker when the $ includes capacity-model cost (not an exact per-token bill).
            billed = f"~${ru.billed_usd:.4f}" if ru.cost_bounded else f"${ru.billed_usd:.4f}"
            btitle = " title='includes BOUNDED capacity-model cost'" if ru.cost_bounded else ""
            out.append(
                f"<tr class='{cls}'><td>{html.escape(k)}</td><td>{ru.calls}</td>"
                f"<td>{ru.input_tokens:,}</td><td>{ru.output_tokens:,}</td>"
                f"<td>{ru.reasoning_tokens:,}</td><td{btitle}>{billed}</td>"
                f"<td>{ru.flagged}</td><td>${ru.overbilled_usd:.4f}</td></tr>"
            )
        return "".join(out)

    # Collect capacity-model notes (e.g. rented utilisation %) across the workload for a legend.
    _cap_notes: list[str] = []
    _any_bounded = False
    for ru in by_prov.values():
        if ru.cost_bounded:
            _any_bounded = True
        for n in ru.cost_notes:
            if n not in _cap_notes:
                _cap_notes.append(n)
    bounded_note = ""
    if _any_bounded:
        items = "".join(f"<li>{html.escape(n)}</li>" for n in _cap_notes)
        bounded_note = (f"<p class='note'><strong>~</strong> {html.escape(_BOUNDED_LEGEND)}</p>"
                        + (f"<ul class='note'>{items}</ul>" if items else ""))

    def cpa_rows(d: dict[str, Rollup]) -> str:
        out = []
        tot_l = tot_u = 0
        for k, ru in sorted(d.items(), key=lambda kv: -kv[1].billed_usd):
            cpa = cost_per_accepted(ru)
            cpa_s = html.escape(cpa) if isinstance(cpa, str) else f"${cpa:.4f}"
            out.append(
                f"<tr><td>{html.escape(k)}</td><td>{ru.accepted_calls}</td>"
                f"<td>{ru.labeled_calls}</td><td>{ru.unlabeled_calls}</td>"
                f"<td>${ru.labeled_billed_usd:.4f}</td><td>{cpa_s}</td></tr>"
            )
            tot_l += ru.labeled_calls; tot_u += ru.unlabeled_calls
        cov = (f"<p class='note'>Cost-per-accepted computed over {tot_l} labeled call(s); "
               f"{tot_u} unlabeled.</p>")
        return "".join(out), cov

    cpa_body, cpa_cov = cpa_rows(by_act)
    cpa_hdr = ("<tr><th>activity</th><th>accepted</th><th>labeled</th><th>unlabeled</th>"
               "<th>labeled $</th><th>$/accepted</th></tr>")

    hdr = "<tr><th>{}</th><th>calls</th><th>input</th><th>output</th><th>reasoning</th><th>billed</th><th>flagged</th><th>overbill</th></tr>"
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>TokenLedger</title>
<style>
 body{{font-family:system-ui,sans-serif;margin:2rem;color:#111;background:#fafafa}}
 h1{{font-size:1.4rem}} h2{{font-size:1.05rem;margin-top:2rem;color:#333}}
 table{{border-collapse:collapse;width:100%;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
 th,td{{padding:.5rem .7rem;text-align:right;border-bottom:1px solid #eee;font-variant-numeric:tabular-nums}}
 th:first-child,td:first-child{{text-align:left}}
 tr.flag td{{background:#fff4f4}} .note{{color:#666;font-size:.85rem;margin:.4rem 0 1.2rem}}
</style></head><body>
<h1>TokenLedger — independent token metering reconciliation</h1>
<p class="note">Self-hosted. Output tokens re-tokenized from returned text (exact where a
public tokenizer exists). Input bounded. Reasoning and cache recorded, not asserted.
Rows highlighted red have an output over-count or out-of-band figure.</p>
<h2>By provider</h2><table>{hdr.format('provider')}{rows(by_prov,'provider')}</table>
{bounded_note}
<h2>By activity type</h2><table>{hdr.format('activity')}{rows(by_act,'activity')}</table>
<h2>Cost per accepted output by activity</h2>
<table>{cpa_hdr}{cpa_body}</table>
{cpa_cov}
<p class="note"><strong>{html.escape(_CPA_CAVEAT)}</strong></p>
<h2>By session</h2><table>{hdr.format('session')}{rows(by_sess,'session')}</table>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path
