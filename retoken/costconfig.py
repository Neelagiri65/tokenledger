"""
Vendor-neutral cost-model config — declare a rented/flat/per-token model in a JSON file so a
partner can register a non-token endpoint WITHOUT writing code. Loaded into core.COST_MODELS.

File shape (retoken-models.json by default):
  {
    "models": {
      "my-rented-llama": {"type": "rented_compute", "usd_per_gpu_hour": 3.5,
                          "gpu_count": 2, "gpu_hours": 720, "throughput_tokens_per_hour": 1200000},
      "vendor-flat":     {"type": "flat_subscription", "fee_per_period": 4000},
      "custom-pt":       {"type": "per_token", "input": 1.0, "output": 3.0,
                          "reasoning": 3.0, "cache_read": 0.1}
    }
  }

No data egress: this only reads/writes a local file and constructs in-process cost models.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .core import (
    COST_MODELS, CostModel, PerTokenCost, FlatSubscriptionCost, RentedComputeCost,
)

DEFAULT_CONFIG = "retoken-models.json"

# Per type: the spec keys allowed (and, for the required ones, no default). Keeping this explicit
# gives clear errors instead of silently dropping a misspelled field.
_PER_TOKEN_KEYS = ("input", "output", "reasoning", "cache_read")
_FLAT_KEYS = ("fee_per_period",)
_RENTED_REQUIRED = ("usd_per_gpu_hour",)
_RENTED_OPTIONAL = ("gpu_count", "gpu_hours", "throughput_tokens_per_hour")


def _nonneg(name: str, v: float) -> float:
    """A cost/quantity must be >= 0 — a negative rate or fee would silently inflate apparent
    savings in the evaluator. Fail loud rather than cost a nonsensical model."""
    if v < 0:
        raise ValueError(f"{name} must be >= 0, got {v}")
    return v


def _positive(name: str, v: float) -> float:
    if v <= 0:
        raise ValueError(f"{name} must be > 0, got {v}")
    return v


def model_from_spec(spec: dict[str, Any]) -> CostModel:
    """Build a CostModel from one config entry. Raises ValueError on an unknown type, a missing
    required field, or a nonsensical value (negative cost, non-positive throughput/gpu_count) —
    fail loud, never cost a model with a silently-wrong shape."""
    t = spec.get("type")
    if t == "per_token":
        missing = [k for k in _PER_TOKEN_KEYS if k not in spec]
        if missing:
            raise ValueError(f"per_token model missing {missing}; need {list(_PER_TOKEN_KEYS)}")
        cc = spec.get("cache_creation")   # optional cache-WRITE rate (default 0)
        return PerTokenCost(
            _nonneg("input", float(spec["input"])), _nonneg("output", float(spec["output"])),
            _nonneg("reasoning", float(spec["reasoning"])),
            _nonneg("cache_read", float(spec["cache_read"])),
            cache_creation=_nonneg("cache_creation", float(cc)) if cc is not None else 0.0,
        )
    if t == "flat_subscription":
        if "fee_per_period" not in spec:
            raise ValueError("flat_subscription model missing 'fee_per_period'")
        return FlatSubscriptionCost(
            fee_per_period=_nonneg("fee_per_period", float(spec["fee_per_period"])),
            label=str(spec.get("label", "")),
        )
    if t == "rented_compute":
        if "usd_per_gpu_hour" not in spec:
            raise ValueError("rented_compute model missing 'usd_per_gpu_hour'")
        gpu_hours = spec.get("gpu_hours")
        tput = spec.get("throughput_tokens_per_hour")
        return RentedComputeCost(
            usd_per_gpu_hour=_nonneg("usd_per_gpu_hour", float(spec["usd_per_gpu_hour"])),
            gpu_count=int(_positive("gpu_count", int(spec.get("gpu_count", 1)))),
            gpu_hours=_nonneg("gpu_hours", float(gpu_hours)) if gpu_hours is not None else None,
            # throughput is a denominator -> must be strictly positive when supplied.
            throughput_tokens_per_hour=(
                _positive("throughput_tokens_per_hour", float(tput)) if tput is not None else None),
        )
    raise ValueError(
        f"unknown cost-model type {t!r}; expected per_token | flat_subscription | rented_compute"
    )


def read_config(path: str = DEFAULT_CONFIG) -> dict[str, dict]:
    """Return the raw {model_id: spec} map from the file, or {} if it does not exist."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    models = data.get("models", {})
    if not isinstance(models, dict):
        raise ValueError(f"{path}: 'models' must be an object of model_id -> spec")
    return models


def load_cost_models(path: str = DEFAULT_CONFIG) -> int:
    """Populate core.COST_MODELS from the config file. Returns the number of models loaded.
    A model already present in COST_MODELS is overwritten by the config (config is the source of
    truth for non-token models). Per-token models in the config also register (overriding PRICING
    for that id). No-op returning 0 if the file is absent."""
    models = read_config(path)
    n = 0
    for model_id, spec in models.items():
        COST_MODELS[model_id] = model_from_spec(spec)   # raises on a bad spec — fail loud
        n += 1
    return n


def add_model(path: str, model_id: str, spec: dict[str, Any]) -> None:
    """Validate a spec (by constructing the model) then read-modify-write it into the config file.
    Creates the file if absent. Validation happens BEFORE the write so a bad spec never persists."""
    model_from_spec(spec)  # validate; raises before we touch the file
    data: dict[str, Any] = {"models": {}}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("models", {})
    data["models"][model_id] = spec
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
