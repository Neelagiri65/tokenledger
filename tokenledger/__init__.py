"""TokenLedger — self-hosted, cross-provider LLM token metering reconciliation."""

from .core import (
    CallRecord, Usage, Confidence, Verdict, BucketResult, CallReconciliation,
    reconcile_call, reconcile_billing_period, count_tokens, PRICING,
    CostModel, PerTokenCost, FlatSubscriptionCost, RentedComputeCost,
    UsageAggregate, CostResult, COST_MODELS, cost_model_for,
)
from .quality import QualitySignal
from .store import Store
from .recorder import record_call, from_openai, from_anthropic
from .dashboard import (
    print_summary, write_html, discrepancy_report_md, reconcile_all, rollup_by,
    cost_per_accepted,
)
from .evaluator import (
    Candidate, CandidateProjection, MigrationReport, BreakEven,
    evaluate_migration, breakeven,
    QUALITY_MEASURED, QUALITY_ASSUMED, QUALITY_UNKNOWN,
)
from .costconfig import (
    load_cost_models, read_config, add_model, model_from_spec, DEFAULT_CONFIG,
)
from .proxy import serve as serve_proxy, run_proxy, capture_and_record

__version__ = "0.1.0"
