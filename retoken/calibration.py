"""
Calibration — overcome the closed-tokenizer blind spot for SYSTEMATIC bias.

The conformance battery (conformance.py) catches instability/drift but not a consistent
multiplier. Calibration closes that gap using TOKENIZER-INVARIANT probes: inputs whose true
token count every reasonable tokenizer agrees on (N whitespace-separated copies of an atomic
token => ~N tokens in tiktoken, Qwen, Mistral, ...). For such inputs there is NO legitimate
"different tokenizer" excuse for a different number, so fitting reported-vs-known recovers any
systematic multiplier/overhead the provider applies.

HONEST SCOPE: this proves the metering pipeline's faithfulness ON THE PROBE CLASS. Generalising
to arbitrary production text assumes the same tokenizer/pipeline handles both (reasonable, but
not cryptographic). Legitimate tokenizer differences DO exist for arbitrary text — that is why we
use invariant probes, not random text, as the yardstick.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

CountFn = Callable[[str], int]


def invariant_probes(unit: str = " the", sizes=(16, 32, 64, 128, 256)) -> list[tuple[str, int]]:
    """(text, known_true_tokens). `unit` is a single atomic token in all common tokenizers."""
    return [(unit * k, k) for k in sizes]


def validate_invariance(probes: list[tuple[str, int]], encoders: dict[str, CountFn],
                        rel_tol: float = 0.05) -> dict[str, bool]:
    """Confirm each trusted open tokenizer counts the probes ~= known true (so the probes really
    are tokenizer-invariant). Run this once to trust your yardstick."""
    out = {}
    for name, fn in encoders.items():
        ok = all(abs(fn(t) - true) <= 1 + rel_tol * true for t, true in probes)
        out[name] = ok
    return out


def _fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / denom if denom else 1.0
    intercept = (sy - slope * sx) / n
    return slope, intercept


@dataclass
class CalibrationResult:
    slope: float          # multiplier the provider applies vs known-true (1.0 = faithful)
    intercept: float      # fixed per-call overhead tokens
    biased: bool
    detail: str


def detect_systematic_bias(count_fn: CountFn, probes: list[tuple[str, int]] | None = None,
                           slope_tol: float = 0.03) -> CalibrationResult:
    probes = probes or invariant_probes()
    xs = [float(true) for _, true in probes]
    ys = [float(count_fn(t)) for t, _ in probes]
    slope, intercept = _fit(xs, ys)
    biased = abs(slope - 1.0) > slope_tol
    pct = (slope - 1.0) * 100
    detail = (f"reported = {slope:.3f} x true + {intercept:.1f}  =>  "
              + (f"SYSTEMATIC {pct:+.1f}% on token count" if biased
                 else "faithful (within tolerance)"))
    return CalibrationResult(slope, intercept, biased, detail)
