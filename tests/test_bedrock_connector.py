"""
Tests for the AWS Bedrock connector (SKELETON against the published ModelInvocationLog schema).
Exercises: modelId -> (provider, model) routing; top-level vs Converse-usage-block token counts;
cache-read fallback; best-effort text extraction (and the honest no-text path); the model-swap
matrix; no data egress; and end-to-end ingest + reconcile of the bundled sample.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenledger.core import Confidence, reconcile_call, Verdict
from tokenledger.store import Store
from tokenledger.connectors.bedrock import (
    parse_bedrock_record, ingest_bedrock_invocation_logs, _provider_model_from_bedrock_id,
    bedrock_model_swap_candidates, BEDROCK_MODEL_SWAP_MATRIX,
)

_SAMPLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "examples", "sample_bedrock_invocation_logs.jsonl")


def _tmp_db() -> str:
    fd, p = tempfile.mkstemp(suffix=".db"); os.close(fd); os.remove(p)
    return p


def test_provider_model_routing():
    assert _provider_model_from_bedrock_id("anthropic.claude-3-5-sonnet-20240620-v1:0") == (
        "anthropic", "anthropic.claude-3-5-sonnet-20240620-v1:0")
    # cross-region prefix stripped for vendor; FULL id kept as model (so 'llama' substring routes)
    prov, model = _provider_model_from_bedrock_id("us.meta.llama3-1-70b-instruct-v1:0")
    assert prov == "meta" and "llama" in model
    assert _provider_model_from_bedrock_id("amazon.nova-pro-v1:0")[0] == "amazon"
    print("PASS test_provider_model_routing")


def test_parse_toplevel_counts_and_text():
    rec = parse_bedrock_record({
        "modelId": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "operation": "InvokeModel", "timestamp": "2026-06-23T10:00:00Z",
        "requestId": "r1", "identity": {"arn": "arn:role/x"},
        "input": {"inputBodyJson": {"messages": [{"role": "user",
                  "content": [{"type": "text", "text": "hello there"}]}]}, "inputTokenCount": 210},
        "output": {"outputBodyJson": {"content": [{"type": "text", "text": "a reply"}]},
                   "outputTokenCount": 160},
    })
    assert rec is not None
    assert rec.provider == "anthropic"
    assert rec.reported.input_tokens == 210 and rec.reported.output_tokens == 160
    assert "hello there" in rec.request_text and "a reply" in rec.response_text
    assert rec.session_id == "r1"
    # central honesty claim: anthropic is closed -> output re-count is BOUNDED, never EXACT
    out_b = [b for b in reconcile_call(rec).buckets if b.bucket == "output"][0]
    assert out_b.confidence == Confidence.BOUNDED
    print("PASS test_parse_toplevel_counts_and_text")


def test_malformed_records_do_not_crash():
    # A connector ingesting external logs must tolerate junk: non-dict input/output, a missing
    # body, and a corrupt JSONL line must skip the record, never abort the ingest.
    assert parse_bedrock_record({"modelId": "amazon.nova-pro-v1:0", "input": "oops",
                                 "output": ["bad"]}) is not None  # coerced, no crash
    dbpath = _tmp_db()
    badfile = dbpath.replace(".db", ".jsonl")
    with open(badfile, "w", encoding="utf-8") as f:
        f.write('{"modelId":"amazon.nova-pro-v1:0","input":{"inputTokenCount":10},"output":{"outputTokenCount":5}}\n')
        f.write('this is not json\n')                                   # corrupt line
        f.write('{"modelId":"anthropic.claude-3-haiku-20240307-v1:0","input":{"inputTokenCount":7},"output":{"outputTokenCount":3}}\n')
    db = Store(dbpath)
    try:
        n = ingest_bedrock_invocation_logs(badfile, db)
        assert n == 2, f"corrupt line must be skipped, 2 good records kept, got {n}"
    finally:
        for p in (dbpath, badfile):
            if os.path.exists(p):
                os.remove(p)
    print("PASS test_malformed_records_do_not_crash")


def test_parse_converse_usage_fallback_and_cache():
    # No top-level token counts -> read from the Converse output body usage block; cache read too.
    rec = parse_bedrock_record({
        "modelId": "us.meta.llama3-1-70b-instruct-v1:0", "operation": "Converse",
        "requestId": "r2", "timestamp": "t",
        "input": {"inputBodyJson": {"messages": [{"role": "user", "content": [{"text": "draft an email"}]}]}},
        "output": {"outputBodyJson": {
            "output": {"message": {"content": [{"text": "Hi team"}]}},
            "usage": {"inputTokens": 95, "outputTokens": 70, "cacheReadInputTokens": 20}}},
    })
    assert rec.reported.input_tokens == 95 and rec.reported.output_tokens == 70
    assert rec.reported.cache_read_tokens == 20
    assert "draft an email" in rec.request_text and "Hi team" in rec.response_text
    print("PASS test_parse_converse_usage_fallback_and_cache")


def test_parse_no_body_is_honest_no_text():
    # Body logging disabled (S3 pointer only) -> counts present, NO text -> output stays unverifiable.
    rec = parse_bedrock_record({
        "modelId": "amazon.nova-pro-v1:0", "operation": "InvokeModel", "requestId": "r3", "timestamp": "t",
        "input": {"inputBodyS3Path": "s3://b/in.json", "inputTokenCount": 500},
        "output": {"outputBodyS3Path": "s3://b/out.json", "outputTokenCount": 300},
    })
    assert rec.request_text == "" and rec.response_text == ""
    out_bucket = [b for b in reconcile_call(rec).buckets if b.bucket == "output"][0]
    assert out_bucket.verdict == Verdict.UNCHECKABLE   # no text -> cannot verify, not an overcount
    print("PASS test_parse_no_body_is_honest_no_text")


def test_missing_modelid_skipped():
    assert parse_bedrock_record({"operation": "InvokeModel"}) is None
    assert parse_bedrock_record("not a dict") is None
    print("PASS test_missing_modelid_skipped")


def test_model_swap_matrix():
    cands = bedrock_model_swap_candidates("anthropic.claude-3-5-sonnet-20240620-v1:0")
    assert cands and all(isinstance(c, str) for c in cands)
    assert bedrock_model_swap_candidates("some.unknown-model") == []
    # matrix is non-empty and points premium -> cheaper in-cloud
    assert "claude-3-haiku" in BEDROCK_MODEL_SWAP_MATRIX
    print("PASS test_model_swap_matrix")


def test_no_network_import_in_module():
    import tokenledger.connectors.bedrock as b
    with open(b.__file__, "r", encoding="utf-8") as f:
        src = f.read()
    for bad in ("import requests", "import urllib", "import httpx", "import socket",
                "import boto3", "http.client"):
        assert bad not in src, f"no-egress: {bad} must not appear (S3 bodies are pointers, not fetched)"
    print("PASS test_no_network_import_in_module")


def test_ingest_sample_end_to_end():
    assert os.path.exists(_SAMPLE), "bundled Bedrock sample missing"
    dbpath = _tmp_db()
    db = Store(dbpath)
    try:
        n = ingest_bedrock_invocation_logs(_SAMPLE, db)
        assert n == 3, f"expected 3 records, got {n}"
        recs = db.all_records()
        providers = {r.provider for r in recs}
        assert {"anthropic", "meta", "amazon"} <= providers
        # the nova row (no text) reconciles as UNVERIFIABLE output, never a false overcount
        nova = [r for r in recs if r.provider == "amazon"][0]
        out_b = [b for b in reconcile_call(nova).buckets if b.bucket == "output"][0]
        assert out_b.verdict == Verdict.UNCHECKABLE
    finally:
        if os.path.exists(dbpath):
            os.remove(dbpath)
    print("PASS test_ingest_sample_end_to_end")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all tests passed")
