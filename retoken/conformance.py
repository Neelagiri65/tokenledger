"""
Metering conformance harness — verify a provider's token COUNTS without their tokenizer.

For closed providers (Claude, Gemini) you cannot re-tokenize to prove the absolute count.
But tokenisation of a fixed input is DETERMINISTIC and LINEAR, so a battery of invariant
tests catches real metering problems with no tokenizer at all:

  - determinism : same input -> identical input_token count across N calls / timestamps
  - drift       : same input later -> still the same count (else tokenizer silently changed)
  - linearity   : count(text*2) ~= 2*count(text)
  - additivity  : count(A+B) ~= count(A)+count(B)
  - monotonic   : longer input -> not fewer tokens

HONEST LIMIT: this catches INCONSISTENCY, not absolute correctness. A provider that applies a
*consistent* bias (e.g. x1.10 on every count) passes every invariant while overcharging. Only a
public tokenizer (OpenAI/open-weight) or a provider-cooperative proof can catch that. Pair this
with exact re-count where possible and with 3-way billing reconciliation.

Usage: supply a `count_fn(text) -> int` that returns the provider's REPORTED input_tokens for
that text (from a real call's usage.input_tokens, or the provider's count-tokens API). The harness
is provider-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

CountFn = Callable[[str], int]

BASE_A = ("The quarterly board report covers revenue, churn, hiring plans and the product "
          "roadmap for the next two quarters across all regions.")
BASE_B = ("Refactor the authentication module, add unit tests for the token refresh path, and "
          "fix the race condition in the session cache.")


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass
class ConformanceReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def __str__(self) -> str:
        lines = [("PASS" if c.passed else "FAIL") + f"  {c.name}: {c.detail}" for c in self.checks]
        lines.append(("ALL PASSED" if self.passed else "DISCREPANCIES FOUND")
                     + " — note: passing does NOT prove absolute correctness on closed tokenizers")
        return "\n".join(lines)


def _within(observed: float, expected: float, abs_slack: int, rel_tol: float) -> bool:
    return abs(observed - expected) <= abs_slack + rel_tol * expected


def run_conformance(count_fn: CountFn, *, base_a: str = BASE_A, base_b: str = BASE_B,
                    runs: int = 4, abs_slack: int = 4, rel_tol: float = 0.03) -> ConformanceReport:
    r = ConformanceReport()

    # determinism: same input, many calls -> identical count
    counts = [count_fn(base_a) for _ in range(runs)]
    a = counts[0]
    det_ok = len(set(counts)) == 1
    r.checks.append(CheckResult(
        "determinism", det_ok,
        f"{runs} calls of identical input -> counts {sorted(set(counts))}"
        + ("" if det_ok else "  <-- same text billed at different token counts")))

    b = count_fn(base_b)

    # monotonicity
    a2 = count_fn(base_a + " " + base_a)
    r.checks.append(CheckResult("monotonic", a2 > a, f"count(A*2)={a2} > count(A)={a}"))

    # linearity (repetition)
    lin_ok = _within(a2, 2 * a, abs_slack, rel_tol)
    r.checks.append(CheckResult(
        "linearity", lin_ok, f"count(A*2)={a2} vs 2*count(A)={2*a}"
        + ("" if lin_ok else "  <-- repetition not ~linear")))

    # additivity
    ab = count_fn(base_a + " " + base_b)
    add_ok = _within(ab, a + b, abs_slack + 2, rel_tol)
    r.checks.append(CheckResult(
        "additivity", add_ok, f"count(A+B)={ab} vs count(A)+count(B)={a+b}"
        + ("" if add_ok else "  <-- A+B != A plus B")))

    return r


def drift_check(count_fn: CountFn, baseline: dict[str, int]) -> ConformanceReport:
    """Compare current counts of known inputs against a stored baseline (run days apart)."""
    r = ConformanceReport()
    for text, was in baseline.items():
        now = count_fn(text)
        ok = now == was
        r.checks.append(CheckResult("drift", ok,
                                    f"baseline={was} now={now}" + ("" if ok else "  <-- tokenizer changed")))
    return r
