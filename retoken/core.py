"""
TokenLedger core: call records, independent token counting, and 3-way reconciliation.

Design constraints (non-negotiable, see README "Architectural constraint test"):
- No data egress. Counting and reconciliation run locally. Nothing in here makes a
  network call. Content can be stored hashed/redacted (see store.py).
- Honest confidence. Every bucket is labelled EXACT, BOUNDED, or UNVERIFIABLE. The tool
  never claims to have proven something it only estimated.
- Multi-provider, multi-session. A record carries provider/model/route/user/session.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional

import os

from .quality import QualitySignal  # provider-agnostic, stdlib-only (no import cycle)

# Exact verification needs the model's REAL tokenizer:
#  - OpenAI family  -> tiktoken (bundled, offline)
#  - open-weight    -> the model's HuggingFace tokenizer (Qwen/Mistral/Llama/DeepSeek/Gemma...)
#  - closed (Anthropic, Gemini) -> NO public tokenizer => estimate only
# Missing libs/tokenizers degrade to an estimator (labelled BOUNDED), never a crash.
try:
    import tiktoken  # type: ignore
    _HAVE_TIKTOKEN = True
except Exception:  # pragma: no cover
    _HAVE_TIKTOKEN = False

try:
    from tokenizers import Tokenizer as _HFTokenizer  # type: ignore
    _HAVE_HF = True
except Exception:  # pragma: no cover
    _HAVE_HF = False

# Providers whose tokenizer is OpenAI's (tiktoken exact).
_OPENAI_FAMILY = {"openai", "azure_openai", "azure"}
# Closed proprietary tokenizers — no public version, estimate only.
_CLOSED_TOKENIZER = {"anthropic", "claude", "gemini", "google", "vertex_ai", "vertexai"}

# model-id (substring, lowercased) -> HuggingFace repo for the exact tokenizer.
# Served model ids vary by provider (NVIDIA "meta/llama-3.1-8b-instruct" vs HF
# "meta-llama/Llama-3.1-8B-Instruct"); this normalises the common open-weight families.
# Override/extend via env TOKENLEDGER_TOKENIZER_MAP='{"substr":"hf/repo"}'.
MODEL_TOKENIZER_REPO: dict[str, str] = {
    "qwen3": "Qwen/Qwen2.5-7B-Instruct", "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3", "mixtral": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama-3": "meta-llama/Llama-3.1-8B-Instruct", "llama3": "meta-llama/Llama-3.1-8B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek": "deepseek-ai/DeepSeek-V2.5", "gemma": "google/gemma-2-9b-it",
}

_hf_cache: dict[str, Any] = {}


def _resolve_hf_repo(model: str) -> Optional[str]:
    m = (model or "").lower()
    import json as _json
    extra = os.environ.get("TOKENLEDGER_TOKENIZER_MAP")
    table = dict(MODEL_TOKENIZER_REPO)
    if extra:
        try:
            table.update({k.lower(): v for k, v in _json.loads(extra).items()})
        except Exception:
            pass
    for key, repo in table.items():
        if key in m:
            return repo
    if "/" in (model or ""):  # already looks like an HF repo id
        return model
    return None


def _hf_encoder(repo: str):
    if repo in _hf_cache:
        return _hf_cache[repo]
    tok = _HFTokenizer.from_pretrained(repo)  # uses HF_TOKEN env for gated repos (e.g. Llama)
    _hf_cache[repo] = tok
    return tok


class Confidence(str, Enum):
    EXACT = "exact"          # re-tokenized with the model's real tokenizer
    BOUNDED = "bounded"      # estimated within a known band
    UNVERIFIABLE = "unverifiable"  # cannot be checked from the response at all


class Verdict(str, Enum):
    OK = "ok"
    OVERCOUNT = "overcount"        # provider billed MORE than we can account for
    UNDERCOUNT = "undercount"      # provider billed LESS (their loss, still worth noting)
    OUT_OF_BAND = "out_of_band"    # estimate-based, reported figure outside tolerance band
    UNCHECKABLE = "uncheckable"    # reasoning / cache: recorded, not judged


# Illustrative pricing, USD per 1M tokens. Replace with your contracted rates.
# (input, output, reasoning, cache_read) — reasoning is billed at the output rate by default.
# This stays the human-editable config for PER-TOKEN models; PerTokenCost objects are derived
# from it. Non-per-token models (flat_subscription / rented_compute) go in COST_MODELS below.
PRICING: dict[str, tuple[float, float, float, float]] = {
    "gpt-4o":            (2.50, 10.00, 10.00, 1.25),
    "gpt-4o-mini":       (0.15,  0.60,  0.60, 0.075),
    "o1":                (15.0, 60.00, 60.00, 7.50),
    "claude-opus-4":     (15.0, 75.00, 75.00, 1.50),
    "claude-sonnet-4":   (3.00, 15.00, 15.00, 0.30),
    "_default":          (1.00,  3.00,  3.00, 0.10),
}


@dataclass
class Usage:
    """What the provider reported — normalised to a CANONICAL DISJOINT model: every billed token is
    counted in exactly ONE bucket, so cost = Σ(bucket × its rate) never double-counts. Providers
    report OVERLAPPING (OpenAI prompt_tokens INCLUDES cached; completion_tokens INCLUDES reasoning);
    the adapters (recorder.from_*, connectors) SUBTRACT the subsets to produce this disjoint shape.
      input_tokens          = uncached input
      output_tokens         = VISIBLE output (excludes reasoning)
      reasoning_tokens      = hidden reasoning/thinking (billed at the output rate)
      cache_read_tokens     = cached input read
      cache_creation_tokens = cache write
    """
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


# --- pluggable cost model -------------------------------------------------------------
#
# Three billing SHAPES, two COMPUTATION MODES:
#   per_token         -> per-call EXACT cost (config rate x counted tokens)      [per-call]
#   flat_subscription -> fixed fee / period, amortised over measured tokens      [capacity]
#   rented_compute    -> provisioned GPU-hours x $/GPU-hour, amortised           [capacity]
#
# The common denominator that makes all three comparable (and is the migration evaluator's
# metric) is EFFECTIVE $/TOKEN = period_cost / measured_tokens.
#
# Cost confidence is ORTHOGONAL to count confidence: a count can be EXACT (re-tokenised)
# while its cost is BOUNDED (rented/flat, depends on a utilisation assumption). per_token
# cost is EXACT (a config rate times a counted number). flat/rented effective cost is
# BOUNDED and is NOT per_token_billable — there is no per-token bill to dispute a dollar
# figure against, so the overbill-$ reconciliation does not apply to it (the count check
# still runs; we just cannot put a dollar on a surplus).

@dataclass
class UsageAggregate:
    """Summed usage over a set of calls (a billing period, a team, an activity bucket)."""
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        # All five disjoint billed buckets (cache_creation included — it is a billed token bucket).
        return (self.input_tokens + self.output_tokens + self.reasoning_tokens
                + self.cache_read_tokens + self.cache_creation_tokens)

    def add_usage(self, u: "Usage") -> None:
        self.input_tokens += u.input_tokens
        self.output_tokens += u.output_tokens
        self.reasoning_tokens += u.reasoning_tokens
        self.cache_read_tokens += u.cache_read_tokens
        self.cache_creation_tokens += u.cache_creation_tokens
        self.calls += 1


@dataclass
class CostResult:
    usd: float
    cost_confidence: Confidence           # EXACT only for per_token; BOUNDED for capacity models
    effective_per_token: Optional[float] = None   # $/token = usd / measured tokens (the denominator)
    per_token_billable: bool = True       # False for flat/rented: no per-token bill to reconcile $ on
    note: str = ""


class CostModel:
    """Base. A cost model turns usage into dollars and the comparable effective $/token."""
    per_token_billable: bool = True

    def period_cost(self, agg: "UsageAggregate") -> CostResult:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class PerTokenCost(CostModel):
    """Pay-per-token: USD per 1M tokens for each DISJOINT bucket. cache_creation (cache-WRITE) is an
    OPTIONAL rate (default 0): built-in PRICING is a 4-tuple so it defaults to 0 — set your contract's
    cache-write rate via config when it applies (e.g. Anthropic ~1.25x input for 5-min writes)."""
    input: float
    output: float
    reasoning: float
    cache_read: float
    cache_creation: float = 0.0
    per_token_billable: bool = True

    def call_cost(self, u: "Usage") -> CostResult:
        usd = (u.input_tokens * self.input
               + u.output_tokens * self.output
               + u.reasoning_tokens * self.reasoning
               + u.cache_read_tokens * self.cache_read
               + u.cache_creation_tokens * self.cache_creation) / 1e6
        toks = (u.input_tokens + u.output_tokens + u.reasoning_tokens
                + u.cache_read_tokens + u.cache_creation_tokens)
        eff = usd / toks if toks else None
        return CostResult(usd, Confidence.EXACT, eff, True, "")

    def period_cost(self, agg: "UsageAggregate") -> CostResult:
        # Linear: equals the sum of per-call costs. EXACT.
        return self.call_cost(Usage(
            agg.input_tokens, agg.output_tokens, agg.reasoning_tokens,
            agg.cache_read_tokens, agg.cache_creation_tokens,
        ))


@dataclass
class FlatSubscriptionCost(CostModel):
    """Fixed fee per period, decoupled from token count. Effective $/token is BOUNDED:
    it amortises the fee over whatever was actually measured in the period."""
    fee_per_period: float
    label: str = ""
    per_token_billable = False  # class attr (no annotation) -> not a dataclass field

    def period_cost(self, agg: "UsageAggregate") -> CostResult:
        toks = agg.total_tokens
        if toks:
            note = (f"flat fee ${self.fee_per_period:,.2f}/period amortised over "
                    f"{toks:,} measured tokens")
            eff = self.fee_per_period / toks
        else:
            note = "flat fee; no measured tokens yet — effective $/token undefined"
            eff = None
        return CostResult(self.fee_per_period, Confidence.BOUNDED, eff, False, note)


@dataclass
class RentedComputeCost(CostModel):
    """Rent the weights on GPU capacity: you pay capacity x time, not tokens. Effective
    $/token is BOUNDED. Supply gpu_hours (provisioned wall-clock hours — PREFERRED, it
    exposes idle capacity) or, if unknown, throughput_tokens_per_hour as an assumed rate."""
    usd_per_gpu_hour: float
    gpu_count: int = 1
    gpu_hours: Optional[float] = None                   # provisioned wall-clock hours (capacity x time)
    throughput_tokens_per_hour: Optional[float] = None  # fallback assumed rate, or for utilisation
    per_token_billable = False

    def period_cost(self, agg: "UsageAggregate") -> CostResult:
        toks = agg.total_tokens
        if self.gpu_hours is not None:
            period = self.usd_per_gpu_hour * self.gpu_count * self.gpu_hours
            eff = period / toks if toks else None
            note = (f"{self.gpu_count}x GPU @ ${self.usd_per_gpu_hour:,.2f}/hr x "
                    f"{self.gpu_hours:g}h = ${period:,.2f} provisioned, amortised over "
                    f"{toks:,} measured tokens")
            if toks and self.throughput_tokens_per_hour:
                measured_tph = toks / self.gpu_hours
                util = measured_tph / self.throughput_tokens_per_hour
                note += (f"; utilisation ~{util * 100:.0f}% of "
                         f"{self.throughput_tokens_per_hour:,.0f} tok/hr capacity")
            return CostResult(period, Confidence.BOUNDED, eff, False, note)
        if self.throughput_tokens_per_hour:
            eff = self.usd_per_gpu_hour * self.gpu_count / self.throughput_tokens_per_hour
            period = eff * toks if toks else 0.0
            note = (f"effective ${eff * 1e6:,.4f}/1M tok from ${self.usd_per_gpu_hour:,.2f}/GPU-hr "
                    f"÷ {self.throughput_tokens_per_hour:,.0f} tok/hr (assumed throughput)")
            return CostResult(period, Confidence.BOUNDED, eff, False, note)
        return CostResult(0.0, Confidence.BOUNDED, None, False,
                          "rented_compute needs gpu_hours or throughput_tokens_per_hour to cost")


# Non-per-token cost models keyed by model id. Populate for rented/subscription deployments
# (e.g. an open-weight model on a dedicated endpoint). Per-token models are derived from
# PRICING and need no entry here. Vendor-neutral config, same as PRICING.
COST_MODELS: dict[str, CostModel] = {}


def cost_model_for(model: str) -> CostModel:
    """Resolve a model id to its cost model. A registered override wins; otherwise the
    model is per-token, built from PRICING (default rates if the model is unlisted)."""
    cm = COST_MODELS.get(model)
    if cm is not None:
        return cm
    return PerTokenCost(*PRICING.get(model, PRICING["_default"]))


@dataclass
class CallRecord:
    provider: str
    model: str
    route: str                      # e.g. /chat/completions, /mistral passthrough
    user_id: str
    session_id: str
    ts: str                         # ISO timestamp, supplied by caller (no clock here)
    reported: Usage                 # provider self-report
    task_class: str = "unclassified"  # activity type: coding/admin/pr/outreach/rag/general/...
    request_text: str = ""          # serialized prompt actually sent
    response_text: str = ""         # generated text actually received
    latency_ms: Optional[float] = None
    request_sha: str = ""
    response_sha: str = ""
    quality: Optional["QualitySignal"] = None  # optional per-call quality (cost-per-accepted)
    # The provider's OWN reported $ for this call, when the source carries it (e.g. LiteLLM `spend`,
    # `response_cost`). This is the ACTUAL charge — read it from their bill, never ask the customer to
    # type rates. For closed providers (no exact re-count) this is the PRIMARY cost figure; we still
    # show our independent token re-count alongside. None when the source doesn't provide it.
    reported_cost_usd: Optional[float] = None

    def __post_init__(self) -> None:
        if self.request_text and not self.request_sha:
            self.request_sha = _sha(self.request_text)
        if self.response_text and not self.response_sha:
            self.response_sha = _sha(self.response_text)


@dataclass
class BucketResult:
    bucket: str
    reported: int
    independent: Optional[int]
    confidence: Confidence
    verdict: Verdict
    note: str = ""


@dataclass
class CallReconciliation:
    record: CallRecord
    buckets: list[BucketResult] = field(default_factory=list)

    @property
    def has_overcount(self) -> bool:
        return any(b.verdict in (Verdict.OVERCOUNT, Verdict.OUT_OF_BAND) for b in self.buckets)

    def billed_cost_usd(self) -> float:
        """Per-call billed cost. EXACT for per-token models. For a capacity model (flat/
        rented) a single call has no standalone cost (the bill is a period figure), so this
        is 0.0 — read the cost at the rollup via cost_model_for(model).period_cost(agg)."""
        return _cost(self.record.model, self.record.reported)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _cost(model: str, u: Usage) -> float:
    """Per-CALL cost in USD. Exact for per-token models. A capacity model (flat/rented) has
    no standalone per-call cost — its bill is a period figure amortised over measured usage —
    so this returns 0.0 for those; use cost_model_for(model).period_cost(agg) at the rollup."""
    cm = cost_model_for(model)
    if isinstance(cm, PerTokenCost):
        return cm.call_cost(u).usd
    return 0.0


# --- independent token counting -------------------------------------------------------

_WORD_RE = re.compile(r"\w+|[^\w\s]")


def _estimate(text: str) -> int:
    pieces = _WORD_RE.findall(text)
    return int(round(len(pieces) * 1.05))


def count_tokens(text: str, provider: str, model: str) -> tuple[int, Confidence]:
    """
    Return (count, confidence). Re-tokenises LOCALLY — the text never leaves the box.
      - OpenAI family  -> tiktoken (EXACT)
      - open-weight    -> the model's real HF tokenizer, run locally (EXACT)
      - closed (Anthropic/Gemini) -> local estimate only (BOUNDED); we do NOT call the
        provider's count-tokens API (that would send your text out AND not be independent).
    Never raises on a missing tokenizer/lib — degrades to BOUNDED.
    """
    if not text:
        return 0, Confidence.EXACT
    p = (provider or "").lower()

    if p in _CLOSED_TOKENIZER:
        return _estimate(text), Confidence.BOUNDED

    # Open-weight model id wins over the provider LABEL: an open-weight model served behind an
    # OpenAI-compatible endpoint (vLLM/Ollama/internal gateway, provider="openai") must be counted
    # with its OWN tokenizer, never tiktoken — otherwise we'd mislabel a wrong count as EXACT.
    repo = _resolve_hf_repo(model)
    if repo:
        if _HAVE_HF:
            try:
                return len(_hf_encoder(repo).encode(text).ids), Confidence.EXACT
            except Exception:
                pass  # gated repo w/o token, offline, or download blocked -> estimate, never tiktoken
        return _estimate(text), Confidence.BOUNDED

    if _HAVE_TIKTOKEN and p in _OPENAI_FAMILY:
        try:
            try:
                enc = tiktoken.encoding_for_model(model)
            except Exception:
                enc = tiktoken.get_encoding("o200k_base")
            return len(enc.encode(text)), Confidence.EXACT
        except Exception:
            pass

    return _estimate(text), Confidence.BOUNDED


# Per-message chat formatting overhead (OpenAI cookbook: ~3 tokens/message + 3 priming).
def message_overhead(num_messages: int) -> int:
    return 3 * num_messages + 3


# Additive cushion on ESTIMATE-based bands (input always; output for closed providers). Absorbs
# estimator wobble + hidden context (tool schemas, system framing) so a small count never
# false-flags OUT_OF_BAND on a figure we only estimate. Over-counts that matter are large vs this.
_BAND_CUSHION = 50


# --- reconciliation -------------------------------------------------------------------

def reconcile_call(
    rec: CallRecord,
    input_tolerance: float = 0.25,
    num_messages: int = 1,
) -> CallReconciliation:
    """
    Compare provider-reported usage against an independent count, bucket by bucket.
    The output bucket is the strong check; input is bounded; reasoning and cache are
    recorded but not judged.
    """
    out = CallReconciliation(record=rec)

    # OUTPUT: re-tokenize the text we actually received.
    # If no response text was captured we CANNOT verify output — that is UNVERIFIABLE, not an
    # over-count. Flagging a missing-text row as overcount is a trust-destroying false positive.
    if not rec.response_text:
        # No captured text → cannot verify output. UNVERIFIABLE, not an over-count. Flagging a
        # missing-text row as overcount is a trust-destroying false positive.
        out.buckets.append(BucketResult(
            "output", rec.reported.output_tokens, None,
            Confidence.UNVERIFIABLE, Verdict.UNCHECKABLE,
            "no response text captured; enable prompt/response logging to verify output tokens",
        ))
    else:
        ind_out, conf_out = count_tokens(rec.response_text, rec.provider, rec.model)
        # output_tokens is canonical VISIBLE output (reasoning is excluded at ingestion and billed in
        # its OWN bucket, judged UNVERIFIABLE below). So compare it directly to the visible re-count.
        reasoning = rec.reported.reasoning_tokens or 0
        _rsfx = f" (+{reasoning} reasoning billed separately, unverifiable)" if reasoning else ""
        billed_out = rec.reported.output_tokens
        if conf_out is Confidence.EXACT:
            if billed_out > ind_out + 1:
                v = Verdict.OVERCOUNT
                note = f"billed {billed_out} output vs {ind_out} re-tokenized from returned text{_rsfx}"
            elif billed_out < ind_out - 1:
                v = Verdict.UNDERCOUNT
                note = f"provider billed fewer output tokens than the returned text contains{_rsfx}"
            else:
                v, note = Verdict.OK, (f"output matches re-count{_rsfx}" if reasoning else "")
        else:
            # Estimate-based (closed provider). Add the same additive cushion as the input band:
            # WITHOUT it, a small output gives a tiny multiplicative band and a normal estimator
            # wobble false-flags OUT_OF_BAND — a trust-destroying false positive (harness, 2026-06-23).
            lo, hi = ind_out * (1 - input_tolerance), ind_out * (1 + input_tolerance) + _BAND_CUSHION
            if billed_out > hi:
                v, note = Verdict.OUT_OF_BAND, f"billed {billed_out}, estimate band [{lo:.0f},{hi:.0f}]{_rsfx}"
            else:
                v, note = Verdict.OK, "estimate only (no exact tokenizer for this provider)"
        out.buckets.append(BucketResult("output", rec.reported.output_tokens, ind_out, conf_out, v, note))

    # INPUT: bounded — but only if we actually have the sent text. No request text → UNVERIFIABLE.
    if not rec.request_text:
        out.buckets.append(BucketResult(
            "input", rec.reported.input_tokens, None,
            Confidence.UNVERIFIABLE, Verdict.UNCHECKABLE,
            "no request text captured; enable prompt logging to bound input tokens",
        ))
    else:
        ind_in_text, conf_in = count_tokens(rec.request_text, rec.provider, rec.model)
        ind_in = ind_in_text + message_overhead(num_messages)
        # input_tokens is canonical UNCACHED input; the request text we re-count is the FULL prompt
        # (cached + uncached). So the billed total to compare is input + cache_read.
        billed_in = rec.reported.input_tokens + rec.reported.cache_read_tokens
        lo, hi = ind_in * (1 - input_tolerance), ind_in * (1 + input_tolerance) + _BAND_CUSHION
        if billed_in > hi:
            v_in = Verdict.OUT_OF_BAND
            note_in = f"billed {billed_in} (incl cache) vs expected ~{ind_in} (band up to {hi:.0f}); check tool schemas / hidden context"
        else:
            v_in, note_in = Verdict.OK, ""
        out.buckets.append(BucketResult("input", billed_in, ind_in, Confidence.BOUNDED, v_in, note_in))

    # REASONING: billed, not returned. Cannot verify from the response. Sanity-flag only.
    if rec.reported.reasoning_tokens:
        note_r = "billed but not returned as text; not verifiable per call (see CoIn research)"
        out.buckets.append(BucketResult(
            "reasoning", rec.reported.reasoning_tokens, None,
            Confidence.UNVERIFIABLE, Verdict.UNCHECKABLE, note_r,
        ))

    # CACHE: hit/miss classification is provider-internal. Recorded, not judged.
    if rec.reported.cache_read_tokens or rec.reported.cache_creation_tokens:
        out.buckets.append(BucketResult(
            "cache_read", rec.reported.cache_read_tokens, None,
            Confidence.UNVERIFIABLE, Verdict.UNCHECKABLE,
            "cache hit/miss is provider-internal; verify behaviourally across many calls",
        ))

    return out


def reconcile_billing_period(
    reported_total: int,
    per_call_sum: int,
    tolerance: int = 0,
) -> tuple[Verdict, str]:
    """
    The third number. Compare the provider's billing/usage-API total for a period against
    the sum of per-call usage you captured. When a provider's own two numbers disagree,
    that is a finding that needs no tokenizer at all.
    """
    diff = reported_total - per_call_sum
    if abs(diff) <= tolerance:
        return Verdict.OK, f"billing total matches captured per-call sum ({per_call_sum})"
    v = Verdict.OVERCOUNT if diff > 0 else Verdict.UNDERCOUNT
    return v, f"billing API total {reported_total} vs captured per-call sum {per_call_sum} (diff {diff:+d})"


def to_dict(obj: Any) -> dict:
    return asdict(obj)
