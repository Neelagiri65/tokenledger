"""
Tests for the wire-level capture sidecar (tokenledger/proxy.py). Deterministic unit tests on the
byte extractors + capture_and_record, plus a real-socket integration test that proves the relay is
PASSIVE (client gets the upstream bytes verbatim; an upstream error still relays and never crashes)
and that the call is recorded + reconciled from what we observed on the wire. All localhost, no keys.
"""
import json
import os
import sys
import tempfile
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenledger.core import Verdict
from tokenledger.store import Store
from tokenledger import proxy as P


def _tmp_db() -> str:
    fd, p = tempfile.mkstemp(suffix=".db"); os.close(fd); os.remove(p)
    return p


# --- unit: extractors -------------------------------------------------------------------

def test_extract_from_json_openai():
    body = json.dumps({
        "model": "gpt-4o",
        "choices": [{"message": {"role": "assistant", "content": "hello from the model"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4},
    }).encode()
    model, usage, text = P.extract_from_json(body)
    assert model == "gpt-4o"
    assert usage["completion_tokens"] == 4
    assert text == "hello from the model"
    print("PASS test_extract_from_json_openai")


def test_extract_from_sse_reassembles_deltas_and_usage():
    chunks = [
        {"model": "gpt-4o", "choices": [{"delta": {"content": "Hel"}}]},
        {"model": "gpt-4o", "choices": [{"delta": {"content": "lo"}}]},
        {"model": "gpt-4o", "choices": [{"delta": {}}], "usage": {"prompt_tokens": 5, "completion_tokens": 1}},
    ]
    raw = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n"
    model, usage, text = P.extract_from_sse(raw)
    assert model == "gpt-4o"
    assert text == "Hello"                       # deltas reassembled from what we relayed
    assert usage.get("completion_tokens") == 1    # usage lifted from the final chunk
    print("PASS test_extract_from_sse_reassembles_deltas_and_usage")


def test_request_text_from_messages():
    req = {"model": "gpt-4o", "messages": [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "summarise the report"}]}
    assert "summarise the report" in P.request_text_from(req)
    print("PASS test_request_text_from_messages")


# --- unit: capture_and_record (passive + records + reconciles) ---------------------------

def test_capture_and_record_recounts_output_from_wire():
    db = _tmp_db(); store = Store(db)
    try:
        req = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}).encode()
        # provider OVER-reports output (99) vs the 4-word response we actually saw on the wire
        resp = json.dumps({
            "model": "gpt-4o",
            "choices": [{"message": {"content": "one two three four"}}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 99},
        }).encode()
        P.capture_and_record(store, "openai", "/v1/chat/completions", req, resp,
                             is_stream=False, latency_ms=12.0)
        recs = store.all_records()
        assert len(recs) == 1
        r = recs[0]
        assert r.model == "gpt-4o"
        assert "one two three four" in r.response_text   # re-counted from the bytes WE saw
        assert r.reported.output_tokens == 99
        from tokenledger.core import reconcile_call
        out = [b for b in reconcile_call(r).buckets if b.bucket == "output"][0]
        assert out.verdict == Verdict.OVERCOUNT          # 99 reported vs ~4 on the wire
    finally:
        if os.path.exists(db):
            os.remove(db)
    print("PASS test_capture_and_record_recounts_output_from_wire")


def test_capture_and_record_is_passive_on_garbage():
    # Malformed request/response bytes must NOT raise (passivity); they simply don't record.
    db = _tmp_db(); store = Store(db)
    try:
        P.capture_and_record(store, "openai", "/v1/x", b"not json", b"also not json",
                             is_stream=False, latency_ms=None)   # must not raise
        # nothing usable -> a record may exist with empty text, but the call must not crash
    finally:
        if os.path.exists(db):
            os.remove(db)
    print("PASS test_capture_and_record_is_passive_on_garbage")


# --- integration: real sockets, relay is passive + records ------------------------------

class _MockUpstream(BaseHTTPRequestHandler):
    """Canned OpenAI-style upstream. /fail -> 500; /v1/chat/completions -> JSON with usage."""
    def log_message(self, *a):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        if self.path == "/fail":
            msg = b'{"error":"boom"}'
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return
        if self.path == "/truncate":
            # LIE about Content-Length then close — the proxy's upstream read raises IncompleteRead.
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "1000")
            self.end_headers()
            self.wfile.write(b"short")  # only 5 of the promised 1000 bytes, then connection closes
            return
        body = json.dumps({
            "model": "gpt-4o",
            "choices": [{"message": {"role": "assistant", "content": "relayed response text here"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 5},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _start(server: ThreadingHTTPServer) -> threading.Thread:
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return t


def test_proxy_relays_verbatim_and_records():
    db = _tmp_db(); store = Store(db)
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _MockUpstream)
    up_port = upstream.server_address[1]
    proxy = P.serve(store, f"http://127.0.0.1:{up_port}", port=0, provider="openai")
    px_port = proxy.server_address[1]
    _start(upstream); _start(proxy)
    try:
        req = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi there"}]}).encode()
        r = urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{px_port}/v1/chat/completions", data=req,
            headers={"Content-Type": "application/json"}), timeout=5)
        got = json.loads(r.read())
        assert got["choices"][0]["message"]["content"] == "relayed response text here"  # verbatim
        # the call is recorded AFTER the relay (in the handler thread) — poll briefly for it.
        import time as _t
        recs = []
        for _ in range(150):
            recs = store.all_records()
            if recs:
                break
            _t.sleep(0.02)
        assert len(recs) == 1 and recs[0].model == "gpt-4o"
        assert "relayed response text here" in recs[0].response_text
        assert recs[0].reported.output_tokens == 5

        # PASSIVITY: an upstream 500 must relay cleanly to the client, never crash the proxy
        try:
            urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{px_port}/fail", data=req,
                headers={"Content-Type": "application/json"}), timeout=5)
            assert False, "expected the relayed 500 to raise HTTPError client-side"
        except urllib.error.HTTPError as e:
            assert e.code == 500
            assert b"boom" in e.read()                 # upstream error body relayed verbatim
    finally:
        proxy.shutdown(); upstream.shutdown()
        if os.path.exists(db):
            os.remove(db)
    print("PASS test_proxy_relays_verbatim_and_records")


def test_concurrency_no_loss_at_moderate_load():
    # Metering integrity under concurrency. SQLite is single-writer; the proxy records per-request in
    # the handler thread, so writes SERIALISE (this is slow, and under SUSTAINED load beyond the
    # connect busy-timeout could raise+swallow -> silent loss; the production fix is a single
    # background writer queue, see docs/design-wire-level-capture.md). At MODERATE load the timeout
    # absorbs contention -> no loss; this test guards that and documents the boundary.
    import time as _t
    db = _tmp_db(); store = Store(db)
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _MockUpstream)
    up_port = upstream.server_address[1]
    proxy = P.serve(store, f"http://127.0.0.1:{up_port}", port=0, provider="openai")
    px_port = proxy.server_address[1]
    _start(upstream); _start(proxy)
    N = 40
    try:
        def hit(i):
            req = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": f"m{i}"}]}).encode()
            urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{px_port}/v1/chat/completions", data=req,
                headers={"Content-Type": "application/json"}), timeout=15).read()
        threads = [threading.Thread(target=hit, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        recorded = 0
        for _ in range(200):                      # drain the post-relay recording
            recorded = len(store.all_records())
            if recorded >= N:
                break
            _t.sleep(0.02)
        assert recorded == N, f"metering loss under concurrency: sent {N}, recorded {recorded}"
    finally:
        proxy.shutdown(); upstream.shutdown()
        if os.path.exists(db):
            os.remove(db)
    print("PASS test_concurrency_no_loss_at_moderate_load")


def test_upstream_truncation_yields_clean_single_response():
    # HARNESS cross-cutting catch: an UPSTREAM mid-body failure (the proxy did not cause) must not
    # inject a 2nd status line into an already-buffered response. With the fix, a non-stream upstream
    # read error yields ONE clean 502 — urllib parses it without choking (a corrupted double-status
    # would raise a parse/BadStatusLine error instead).
    db = _tmp_db(); store = Store(db)
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _MockUpstream)
    up_port = upstream.server_address[1]
    proxy = P.serve(store, f"http://127.0.0.1:{up_port}", port=0, provider="openai")
    px_port = proxy.server_address[1]
    _start(upstream); _start(proxy)
    try:
        req = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}).encode()
        got_clean_502 = False
        try:
            urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{px_port}/truncate", data=req,
                headers={"Content-Type": "application/json"}), timeout=5).read()
        except urllib.error.HTTPError as e:
            got_clean_502 = (e.code == 502)        # a single, parseable error response
        except Exception:
            got_clean_502 = False                  # a corrupted/double response would land here
        assert got_clean_502, "upstream truncation must yield ONE clean 502, not a corrupted response"
    finally:
        proxy.shutdown(); upstream.shutdown()
        if os.path.exists(db):
            os.remove(db)
    print("PASS test_upstream_truncation_yields_clean_single_response")


def test_no_third_party_egress_in_module():
    with open(P.__file__, "r", encoding="utf-8") as f:
        src = f.read()
    # urllib is REQUIRED to forward to the operator's upstream; third-party SDKs / sockets are not.
    for bad in ("import requests", "import httpx", "import boto3", "aiohttp"):
        assert bad not in src, f"no third-party egress: {bad} must not appear"
    # no hardcoded external host — the only target is the configured upstream
    assert "http://api." not in src and "https://api." not in src
    print("PASS test_no_third_party_egress_in_module")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all tests passed")
