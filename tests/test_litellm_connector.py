"""
Tests for the LiteLLM connector — the DEFAULT ingest path (`retoken ingest --format litellm`),
which shipped a fixture but no test (coverage gap flagged by the harness review). Covers field
mapping, the cache_hit fallback, defensive skips, and end-to-end ingest of the bundled sample with
the honest no-text -> UNVERIFIABLE behaviour.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retoken.core import Verdict, reconcile_call
from retoken.store import Store
from retoken.connectors.litellm import parse_litellm_row, ingest_litellm_spendlog

_SAMPLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "examples", "sample_litellm_spendlog.jsonl")


def _tmp_db() -> str:
    fd, p = tempfile.mkstemp(suffix=".db"); os.close(fd); os.remove(p)
    return p


def test_parse_maps_core_fields_and_text():
    row = {
        "request_id": "r1", "custom_llm_provider": "openai", "model": "gpt-4o",
        "user": "alice", "startTime": "2026-06-21T10:00:00Z",
        "prompt_tokens": 26, "completion_tokens": 64,
        "messages": [{"role": "user", "content": "explain mitochondria"}],
        "response": {"choices": [{"message": {"content": "the powerhouse of the cell"}}]},
    }
    rec = parse_litellm_row(row)
    assert rec.provider == "openai" and rec.model == "gpt-4o"
    assert rec.reported.input_tokens == 26 and rec.reported.output_tokens == 64
    assert rec.user_id == "alice" and rec.session_id == "r1"
    assert "mitochondria" in rec.request_text
    assert "powerhouse" in rec.response_text
    print("PASS test_parse_maps_core_fields_and_text")


def test_cache_hit_fallback_sets_cache_read():
    # No explicit cache_read field, but cache_hit True -> treat prompt tokens as cache-read.
    rec = parse_litellm_row({
        "request_id": "r2", "custom_llm_provider": "anthropic", "model": "claude-sonnet-4",
        "prompt_tokens": 40, "completion_tokens": 10, "cache_hit": True,
    })
    assert rec.reported.cache_read_tokens == 40
    print("PASS test_cache_hit_fallback_sets_cache_read")


def test_explicit_cache_read_preferred():
    rec = parse_litellm_row({
        "model": "gpt-4o", "prompt_tokens": 100, "completion_tokens": 5,
        "cache_read_input_tokens": 30, "cache_hit": True,
    })
    assert rec.reported.cache_read_tokens == 30   # explicit value wins over the hit-fallback
    print("PASS test_explicit_cache_read_preferred")


def test_row_without_model_skipped():
    assert parse_litellm_row({"prompt_tokens": 5}) is None
    print("PASS test_row_without_model_skipped")


def test_captures_provider_reported_cost():
    # The provider's OWN $ (LiteLLM `spend`/`response_cost`) is captured — no typed rates needed.
    rec = parse_litellm_row({"model": "gpt-4o", "prompt_tokens": 26, "completion_tokens": 64,
                             "spend": 0.00071})
    assert rec.reported_cost_usd == 0.00071
    # response_cost alias + bad value tolerated
    assert parse_litellm_row({"model": "x", "response_cost": "0.5"}).reported_cost_usd == 0.5
    assert parse_litellm_row({"model": "x", "spend": "n/a"}).reported_cost_usd is None
    assert parse_litellm_row({"model": "x"}).reported_cost_usd is None
    print("PASS test_captures_provider_reported_cost")


def test_reported_cost_roundtrips_and_surfaces_in_rollup():
    from retoken.dashboard import rollup_by, reconcile_all
    db = _tmp_db(); store = Store(db)
    try:
        n = ingest_litellm_spendlog(_SAMPLE, store)
        assert n == 4
        recs = store.all_records()
        # the bundled sample carries `spend` on at least the first rows -> round-trips through SQLite
        assert any(r.reported_cost_usd is not None for r in recs)
        ru = rollup_by(reconcile_all(store), "provider")["openai"]
        assert ru.reported_cost_calls >= 1
        assert ru.provider_reported_usd > 0
    finally:
        if os.path.exists(db):
            os.remove(db)
    print("PASS test_reported_cost_roundtrips_and_surfaces_in_rollup")


def test_ingest_sample_end_to_end():
    assert os.path.exists(_SAMPLE), "bundled LiteLLM sample missing"
    db = _tmp_db(); store = Store(db)
    try:
        n = ingest_litellm_spendlog(_SAMPLE, store)
        assert n == 4, f"expected 4 rows, got {n}"
        recs = store.all_records()
        assert {"gpt-4o", "o1", "claude-sonnet-4"} <= {r.model for r in recs}
        # the o1 row has no captured text -> output is UNVERIFIABLE, never a false overcount
        o1 = [r for r in recs if r.model == "o1"][0]
        out = next(b for b in reconcile_call(o1).buckets if b.bucket == "output")
        assert out.verdict == Verdict.UNCHECKABLE
    finally:
        if os.path.exists(db):
            os.remove(db)
    print("PASS test_ingest_sample_end_to_end")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all tests passed")
