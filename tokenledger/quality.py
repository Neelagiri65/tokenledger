"""
Quality signal per call — the bridge from cost to cost-PER-OUTCOME.

A call's cost is only half the FinOps picture; the other half is what the spend actually bought.
This module captures an OPTIONAL, customer-supplied quality signal alongside each call so the
dashboard can report cost-per-ACCEPTED-output by activity.

Honest-confidence rules baked in (mirroring core's labelling discipline):
  - None everywhere = "not captured" = UNKNOWN. Absence of a signal is NEVER read as a reject,
    the direct analog of "no captured text -> UNVERIFIABLE, not an over-count".
  - 'accept' is the SOLE driver of the accepted count. eval_score and success are captured and
    displayed separately but do NOT silently OR into accepted (no hidden coupling).

The signal is provider-agnostic: it comes from the customer's own eval/accept events, never a
provider API. Pure local, stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

_STATUSES = ("accept", "reject", "unknown")


def _clamp_score(score: Optional[float]) -> Optional[float]:
    """Clamp an eval score to [0.0, 1.0]. None stays None (not captured)."""
    if score is None:
        return None
    s = float(score)
    if s < 0.0:
        return 0.0
    if s > 1.0:
        return 1.0
    return s


def _coerce_status(status: Optional[str]) -> Optional[str]:
    """Normalise status to {'accept','reject','unknown'}. None stays None; any other string
    normalises to 'unknown' (never silently dropped, never raised)."""
    if status is None:
        return None
    s = str(status).strip().lower()
    return s if s in _STATUSES else "unknown"


def _coerce_success(success: Optional[bool]) -> Optional[bool]:
    """Downstream-success flag -> Optional[bool]. None stays None (not captured)."""
    if success is None:
        return None
    return bool(success)


@dataclass
class QualitySignal:
    """An optional, customer-supplied per-call quality signal.

    All three fields default to None = "not captured" = unknown. Values are normalised on
    construction: eval_score is clamped to [0,1] (not rejected), status coerces to one of
    {'accept','reject','unknown'} ('accept' alone drives the accepted count), success -> bool.
    """

    eval_score: Optional[float] = None
    status: Optional[str] = None
    success: Optional[bool] = None

    def __post_init__(self) -> None:
        self.eval_score = _clamp_score(self.eval_score)
        self.status = _coerce_status(self.status)
        self.success = _coerce_success(self.success)

    def is_empty(self) -> bool:
        """True when nothing was captured (all three None) — treated as no signal."""
        return self.eval_score is None and self.status is None and self.success is None

    def is_labeled(self) -> bool:
        """True when the call carries a known accept/reject label (drives the metric's coverage)."""
        return self.status in ("accept", "reject")

    def is_accepted(self) -> bool:
        """True only when status == 'accept' — the SOLE accepted-count driver."""
        return self.status == "accept"
