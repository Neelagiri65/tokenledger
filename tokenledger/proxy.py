"""
Wire-level capture sidecar (independence ladder, rung 2). A passthrough reverse-proxy that observes
the ACTUAL request/response bytes on the wire and reconciles the provider's reported usage against
our own re-count — independent of any provider log. See docs/design-wire-level-capture.md.

NON-NEGOTIABLES (see the design doc):
  1. PASSIVE — the client always gets the upstream's exact status/headers/body; capture + record run
     only AFTER the response is fully relayed, and every step is swallowed on failure. A bug here
     never breaks or stalls the real call.
  2. NO EGRESS to third parties — the ONLY outbound connection is the customer-configured upstream
     (that is the request the client made). No telemetry, no third-party calls.
  3. STREAMING-AWARE — SSE responses are relayed chunk-by-chunk while a copy is accumulated.
  4. HONEST LABELS — re-count confidence + verdicts come from core; we invent no provider numbers.

Prototype: stdlib http.server + urllib; OpenAI/Anthropic response shapes via the recorder adapters;
testable fully offline against a local mock upstream. Production hardening (concurrency/TLS/limits)
is flagged in the design doc, not built.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from time import perf_counter
from typing import Any, Callable, Optional

from .recorder import record_call
from .store import Store

# Hop-by-hop / length headers we must not forward verbatim (urllib/our relay set their own).
_DROP_REQUEST_HEADERS = {"host", "content-length", "connection", "accept-encoding"}
_DROP_RESPONSE_HEADERS = {"content-length", "transfer-encoding", "connection", "content-encoding"}
_STREAM_CHUNK = 8192


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- text/usage extraction (best-effort; '' when not recoverable -> honest UNVERIFIABLE) -------

def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # [{type,text}] blocks
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""


def request_text_from(req_obj: dict) -> str:
    msgs = req_obj.get("messages")
    if isinstance(msgs, list):
        parts = []
        for m in msgs:
            if isinstance(m, dict):
                parts.append(_content_to_text(m.get("content")))
        return "\n".join(p for p in parts if p)
    for k in ("prompt", "input"):
        if isinstance(req_obj.get(k), str):
            return req_obj[k]
    return ""


def extract_from_json(body: bytes) -> tuple[Optional[str], dict, str]:
    """(model, usage_dict, response_text) from a non-streaming JSON response."""
    obj = json.loads(body)
    model = obj.get("model")
    usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else {}
    text = ""
    choices = obj.get("choices")
    if isinstance(choices, list):  # OpenAI chat
        text = "\n".join(_content_to_text((c.get("message") or {}).get("content"))
                         for c in choices if isinstance(c, dict))
    if not text and isinstance(obj.get("content"), list):  # Anthropic messages
        text = _content_to_text(obj["content"])
    return model, usage, text


def extract_from_sse(raw: bytes) -> tuple[Optional[str], dict, str]:
    """(model, usage_dict, response_text) reassembled from an SSE stream we relayed."""
    model: Optional[str] = None
    usage: dict = {}
    parts: list[str] = []
    for line in raw.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[len(b"data:"):].strip()
        if not data or data == b"[DONE]":
            continue
        try:
            obj = json.loads(data)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        model = model or obj.get("model")
        if isinstance(obj.get("usage"), dict):
            usage.update(obj["usage"])               # OpenAI final chunk / Anthropic message_delta
        for c in (obj.get("choices") or []):
            if isinstance(c, dict):
                parts.append(_content_to_text((c.get("delta") or {}).get("content")))
        d = obj.get("delta")                          # Anthropic content_block_delta
        if isinstance(d, dict) and isinstance(d.get("text"), str):
            parts.append(d["text"])
    return model, usage, "".join(p for p in parts if p)


def capture_and_record(
    store: Store, provider: str, route: str,
    request_body: bytes, response_body: bytes, is_stream: bool,
    latency_ms: Optional[float], on_record: Optional[Callable] = None,
) -> None:
    """Re-count the relayed bytes and record + reconcile. Wrapped so it NEVER raises into the relay
    path (passivity). Best-effort: if we can't parse, we simply don't record that call."""
    try:
        try:
            req_obj = json.loads(request_body) if request_body else {}
        except Exception:
            req_obj = {}
        if is_stream:
            model, usage, resp_text = extract_from_sse(response_body)
        else:
            model, usage, resp_text = extract_from_json(response_body)
        model = model or req_obj.get("model") or "unknown"
        rec = record_call(
            store,
            provider=provider,
            model=str(model),
            user_id=str(req_obj.get("user", "wire")),
            session_id=str(req_obj.get("tl_session_id", "wire")),
            ts=_now(),
            usage=usage or {},                       # raw provider usage -> recorder adapter normalises
            request_text=request_text_from(req_obj),
            response_text=resp_text,
            route=route,
            latency_ms=latency_ms,
        )
        if on_record is not None:
            try:
                on_record(rec)
            except Exception:
                pass
    except Exception:
        pass  # passive: capture failure must never affect the already-relayed response


def make_handler(store: Store, upstream: str, provider: str = "openai",
                 on_record: Optional[Callable] = None):
    upstream = upstream.rstrip("/")

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # silence default stderr logging
            pass

        def _relay(self) -> None:
            path = self.path
            length = int(self.headers.get("Content-Length") or 0)
            req_body = self.rfile.read(length) if length else b""

            fwd_headers = {k: v for k, v in self.headers.items()
                           if k.lower() not in _DROP_REQUEST_HEADERS}
            url = upstream + path
            req = urllib.request.Request(url, data=req_body, headers=fwd_headers,
                                         method=self.command)

            t0 = perf_counter()
            # Open upstream. An HTTP error still carries a real response we must relay verbatim.
            try:
                resp = urllib.request.urlopen(req)  # noqa: S310 - URL is the operator-set upstream
            except urllib.error.HTTPError as e:
                resp = e
            except Exception as e:
                # Upstream unreachable: report a gateway error to the client; nothing to record.
                self.send_error(502, f"upstream error: {e}")
                return

            status = resp.status if hasattr(resp, "status") else resp.getcode()
            resp_headers = [(k, v) for k, v in resp.headers.items()
                            if k.lower() not in _DROP_RESPONSE_HEADERS]
            ctype = (resp.headers.get("Content-Type") or "").lower()
            is_stream = "text/event-stream" in ctype

            captured = bytearray()
            client_alive = True
            if is_stream:
                # Headers go out BEFORE the body streams. Once they're on the wire we must NEVER
                # inject a second status line — so an UPSTREAM read failure mid-stream just stops
                # and closes; it does not become a 502 (that would corrupt the relayed response).
                self.send_response(status)
                for k, v in resp_headers:
                    self.send_header(k, v)
                self.send_header("Content-Type", resp.headers.get("Content-Type", "text/event-stream"))
                self.send_header("Connection", "close")
                self.end_headers()
                self._headers_sent = True
                while True:
                    try:
                        chunk = resp.read(_STREAM_CHUNK)
                    except Exception:
                        break  # upstream failed mid-stream (not our fault) — stop; headers already out
                    if not chunk:
                        break
                    captured.extend(chunk)
                    if client_alive:
                        try:
                            self.wfile.write(chunk)
                            self.wfile.flush()
                        except Exception:
                            client_alive = False  # client gone; keep draining upstream for the record
            else:
                # Read the FULL upstream body BEFORE sending anything, so an upstream read error
                # (IncompleteRead / reset the proxy did not cause) yields a CLEAN 502 instead of a
                # second status line injected after an already-buffered 200 (a passivity violation
                # the harness cross-cutting review proved).
                try:
                    body = resp.read()
                except Exception as e:
                    self.send_error(502, f"upstream read error: {e}")
                    return
                captured.extend(body)
                self.send_response(status)
                for k, v in resp_headers:
                    self.send_header(k, v)
                self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self._headers_sent = True
                try:
                    self.wfile.write(body)
                except Exception:
                    client_alive = False
            latency_ms = (perf_counter() - t0) * 1000.0

            # AFTER the response is fully relayed: capture + record, never affecting the client.
            capture_and_record(store, provider, path, req_body, bytes(captured),
                               is_stream, latency_ms, on_record)

        def do_POST(self):  # noqa: N802
            self._headers_sent = False
            try:
                self._relay()
            except Exception as e:
                # Last-resort guard. Only emit a 502 if NOTHING has been sent yet — once headers are
                # on the wire, a second status line would corrupt the response; just stop.
                if not getattr(self, "_headers_sent", False):
                    try:
                        self.send_error(502, f"proxy error: {e}")
                    except Exception:
                        pass

    return _Handler


def serve(store: Store, upstream: str, port: int = 8088, provider: str = "openai",
          host: str = "127.0.0.1", on_record: Optional[Callable] = None) -> ThreadingHTTPServer:
    """Create (but do not block on) a threaded proxy server. Call .serve_forever() to run, or use
    in tests with a background thread + .shutdown(). Binds localhost by default."""
    handler = make_handler(store, upstream, provider, on_record)
    return ThreadingHTTPServer((host, port), handler)


def run_proxy(db: str, upstream: str, port: int = 8088, provider: str = "openai",
              host: str = "127.0.0.1") -> None:
    """Blocking entry point for the CLI."""
    store = Store(db)
    httpd = serve(store, upstream, port, provider, host)
    print(f"TokenLedger wire-level proxy on http://{host}:{port}  ->  {upstream}  "
          f"(provider={provider}, db={db})")
    print("point your app's base URL at the address above; Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
