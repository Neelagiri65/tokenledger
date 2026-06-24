"""
Tests for the cost-model config (tokenledger/costconfig.py) and the dashboard's BOUNDED-cost
surfacing. Non-negotiables exercised:
  - config builds the right cost model per type; a bad/unknown spec FAILS LOUD (never silently
    costs a model with the wrong shape);
  - add_model validates BEFORE writing (a bad spec never persists);
  - load_cost_models populates COST_MODELS and is a no-op when the file is absent;
  - the dashboard marks a capacity-model cost as BOUNDED (cost_bounded) and carries its note.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenledger.core import (
    COST_MODELS, CallRecord, Usage, PerTokenCost, FlatSubscriptionCost, RentedComputeCost,
)
from tokenledger.costconfig import (
    model_from_spec, read_config, load_cost_models, add_model, DEFAULT_CONFIG,
)
from tokenledger.store import Store
from tokenledger.dashboard import rollup_by, reconcile_all


def _tmp_json() -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(path)  # absent so add_model/load see "no file"
    return path


def _tmp_db() -> str:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)
    return path


# --- model_from_spec --------------------------------------------------------------------

def test_model_from_spec_each_type():
    pt = model_from_spec({"type": "per_token", "input": 1.0, "output": 3.0,
                          "reasoning": 3.0, "cache_read": 0.1})
    assert isinstance(pt, PerTokenCost) and pt.output == 3.0
    flat = model_from_spec({"type": "flat_subscription", "fee_per_period": 4000})
    assert isinstance(flat, FlatSubscriptionCost) and flat.fee_per_period == 4000
    rent = model_from_spec({"type": "rented_compute", "usd_per_gpu_hour": 3.5,
                            "gpu_count": 2, "gpu_hours": 720})
    assert isinstance(rent, RentedComputeCost) and rent.gpu_count == 2 and rent.gpu_hours == 720
    print("PASS test_model_from_spec_each_type")


def test_bad_spec_fails_loud():
    for bad in (
        {"type": "nonsense"},
        {"type": "per_token", "input": 1.0},                 # missing output/reasoning/cache_read
        {"type": "flat_subscription"},                       # missing fee
        {"type": "rented_compute", "gpu_count": 2},          # missing usd_per_gpu_hour
    ):
        try:
            model_from_spec(bad)
            assert False, f"expected ValueError for {bad}"
        except ValueError:
            pass
    print("PASS test_bad_spec_fails_loud")


# --- add_model / read_config / load_cost_models -----------------------------------------

def test_add_model_validates_before_write():
    path = _tmp_json()
    try:
        # a bad spec must NOT create/modify the file
        try:
            add_model(path, "x", {"type": "rented_compute"})  # missing usd_per_gpu_hour
            assert False, "expected ValueError"
        except ValueError:
            pass
        assert not os.path.exists(path), "bad spec must not persist"
        # a good spec writes
        add_model(path, "my-rented", {"type": "rented_compute", "usd_per_gpu_hour": 3.0,
                                      "gpu_count": 1, "gpu_hours": 10})
        assert os.path.exists(path)
        models = read_config(path)
        assert "my-rented" in models and models["my-rented"]["type"] == "rented_compute"
    finally:
        if os.path.exists(path):
            os.remove(path)
    print("PASS test_add_model_validates_before_write")


def test_load_cost_models_populates_and_absent_is_noop():
    assert load_cost_models("/nonexistent-cfg-xyz.json") == 0   # absent -> no-op, no raise
    path = _tmp_json()
    try:
        add_model(path, "flat-vendor", {"type": "flat_subscription", "fee_per_period": 1000})
        n = load_cost_models(path)
        assert n == 1
        assert isinstance(COST_MODELS.get("flat-vendor"), FlatSubscriptionCost)
    finally:
        COST_MODELS.pop("flat-vendor", None)
        if os.path.exists(path):
            os.remove(path)
    print("PASS test_load_cost_models_populates_and_absent_is_noop")


# --- dashboard BOUNDED surfacing --------------------------------------------------------

def test_dashboard_marks_capacity_cost_bounded_with_note():
    COST_MODELS["rented-x"] = RentedComputeCost(usd_per_gpu_hour=2.0, gpu_count=1, gpu_hours=24.0,
                                                throughput_tokens_per_hour=1_000_000)
    dbpath = _tmp_db()
    db = Store(dbpath)
    try:
        db.record(CallRecord(provider="openai", model="rented-x", route="/v1", user_id="u",
                             session_id="s", ts="2026-06-23T00:00:00Z",
                             reported=Usage(input_tokens=500, output_tokens=500),
                             request_text="hi", response_text="there"))
        recs = reconcile_all(db)
        ru = rollup_by(recs, "provider")["openai"]
        assert ru.cost_bounded is True
        assert ru.billed_usd > 0
        assert any("provisioned" in n or "utilisation" in n for n in ru.cost_notes)
    finally:
        COST_MODELS.pop("rented-x", None)
        if os.path.exists(dbpath):
            os.remove(dbpath)
    print("PASS test_dashboard_marks_capacity_cost_bounded_with_note")


def test_per_token_only_rollup_not_bounded():
    dbpath = _tmp_db()
    db = Store(dbpath)
    try:
        db.record(CallRecord(provider="openai", model="gpt-4o", route="/v1", user_id="u",
                             session_id="s", ts="2026-06-23T00:00:00Z",
                             reported=Usage(input_tokens=10, output_tokens=5),
                             request_text="hi", response_text="there"))
        ru = rollup_by(reconcile_all(db), "provider")["openai"]
        assert ru.cost_bounded is False
        assert ru.cost_notes == []
    finally:
        if os.path.exists(dbpath):
            os.remove(dbpath)
    print("PASS test_per_token_only_rollup_not_bounded")


def test_model_rejects_negative_and_nonpositive():
    # Review catch: a negative rate/fee would silently inflate apparent savings; throughput is a
    # denominator so must be > 0. All must FAIL LOUD.
    for bad in (
        {"type": "per_token", "input": -1.0, "output": 3.0, "reasoning": 3.0, "cache_read": 0.1},
        {"type": "flat_subscription", "fee_per_period": -1000},
        {"type": "rented_compute", "usd_per_gpu_hour": -10.0, "gpu_count": 1, "gpu_hours": 24},
        {"type": "rented_compute", "usd_per_gpu_hour": 2.0, "gpu_count": 0, "gpu_hours": 24},
        {"type": "rented_compute", "usd_per_gpu_hour": 2.0, "throughput_tokens_per_hour": 0},
    ):
        try:
            model_from_spec(bad)
            assert False, f"expected ValueError for {bad}"
        except ValueError:
            pass
    print("PASS test_model_rejects_negative_and_nonpositive")


def test_dashboard_marks_flat_and_mixed_bucket_bounded():
    # Review catch: flat model in the dashboard, AND a bucket MIXING per-token + capacity must be
    # marked bounded (the weakest confidence wins).
    COST_MODELS["flat-vendor"] = FlatSubscriptionCost(fee_per_period=1000)
    dbpath = _tmp_db()
    db = Store(dbpath)
    try:
        # same provider bucket: one per-token (gpt-4o) + one flat-vendor call
        db.record(CallRecord(provider="mix", model="gpt-4o", route="/v1", user_id="u",
                             session_id="s1", ts="2026-06-23T00:00:00Z",
                             reported=Usage(input_tokens=100, output_tokens=50),
                             request_text="hi", response_text="there"))
        db.record(CallRecord(provider="mix", model="flat-vendor", route="/v1", user_id="u",
                             session_id="s2", ts="2026-06-23T00:00:00Z",
                             reported=Usage(input_tokens=400, output_tokens=600),
                             request_text="hi", response_text="there"))
        ru = rollup_by(reconcile_all(db), "provider")["mix"]
        assert ru.cost_bounded is True                     # mixed bucket -> bounded
        assert any("flat" in n for n in ru.cost_notes)
        assert ru.billed_usd > 1000                        # flat $1000 + the per-token portion
    finally:
        COST_MODELS.pop("flat-vendor", None)
        if os.path.exists(dbpath):
            os.remove(dbpath)
    print("PASS test_dashboard_marks_flat_and_mixed_bucket_bounded")


def test_html_renders_bounded_marker_and_legend():
    # Review catch: assert the RENDERING, not just internal state — a bug dropping '~' must fail.
    from tokenledger.dashboard import write_html
    COST_MODELS["rented-h"] = RentedComputeCost(usd_per_gpu_hour=10, gpu_count=1, gpu_hours=24)
    dbpath = _tmp_db()
    htmlpath = dbpath.replace(".db", ".html")
    db = Store(dbpath)
    try:
        db.record(CallRecord(provider="local", model="rented-h", route="/v1", user_id="u",
                             session_id="s", ts="2026-06-23T00:00:00Z",
                             reported=Usage(input_tokens=500, output_tokens=500),
                             request_text="hi", response_text="there"))
        write_html(db, htmlpath)
        with open(htmlpath, "r", encoding="utf-8") as f:
            doc = f.read()
        assert "~$" in doc                                 # bounded marker rendered
        assert "BOUNDED capacity-model" in doc             # legend present
    finally:
        COST_MODELS.pop("rented-h", None)
        for p in (dbpath, htmlpath):
            if os.path.exists(p):
                os.remove(p)
    print("PASS test_html_renders_bounded_marker_and_legend")


def test_cli_add_missing_per_token_field_rejected():
    # Review catch: per_token rejection (was only rented tested). Missing --reasoning must fail.
    from tokenledger.cli import main
    path = _tmp_json()
    try:
        rc = main(["models", "add", "incomplete", "--type", "per_token",
                   "--input", "0.03", "--output", "0.06", "--cache_read", "0.015",
                   "--config", path])  # no --reasoning
        assert rc == 1
        assert not os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.remove(path)
    print("PASS test_cli_add_missing_per_token_field_rejected")


def test_cli_add_missing_required_field_rejected():
    # RUN-AND-OBSERVE catch: argparse must NOT default a required field to 0.0 and silently accept
    # an incomplete model. `models add bad --type rented_compute` (no usd_per_gpu_hour) must FAIL
    # and write nothing.
    from tokenledger.cli import main
    path = _tmp_json()
    try:
        rc = main(["models", "add", "bad", "--type", "rented_compute", "--config", path])
        assert rc == 1, "incomplete rented spec must be rejected (rc=1)"
        assert not os.path.exists(path), "rejected spec must not persist"
        # a complete one succeeds
        rc2 = main(["models", "add", "ok", "--type", "rented_compute",
                    "--usd_per_gpu_hour", "3.0", "--gpu_hours", "10", "--config", path])
        assert rc2 == 0 and os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.remove(path)
    print("PASS test_cli_add_missing_required_field_rejected")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all tests passed")
