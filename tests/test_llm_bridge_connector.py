"""
Tests for the llm_bridge metering wrapper (tokenledger/connectors/llm_bridge.py).

These run with NO llm_bridge installed: the wrapper is exercised against a duck-typed fake
client that mimics llm_bridge's LLMClient / LLMResponse contract. That is deliberate — the
connector must work against anything exposing the same tiny interface, and the tests must not
depend on an optional package.

What is asserted (the advisor's four risks):
  1. complete() is metered too (single-turn calls must not bypass the wrapper).
  2. The output re-tokenisation strong-check fires end-to-end through the wrapper (OVERCOUNT).
  3. Passive: a store whose record() raises must NOT break the caller's call.
  4. No egress / raw enrichment is local: reasoning+cache are lifted from resp.raw with no I/O.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenledger.core import Verdict, count_tokens  # noqa: E402
from tokenledger.quality import QualitySignal  # noqa: E402
from tokenledger.store import Store  # noqa: E402
from tokenledger.connectors import metered, MeteringLLMClient  # noqa: E402

RESP_TEXT = "The quick brown fox jumps over the lazy dog. " * 8
_FIXED_TS = lambda: "2026-06-22T00:00:00+00:00"  # noqa: E731 - deterministic, no clock in tests


class FakeResponse:
    """Mimics llm_bridge.LLMResponse."""

    def __init__(self, content, model, prompt_tokens, completion_tokens, latency_ms=12.0, raw=None):
        self.content = content
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.latency_ms = latency_ms
        self.raw = raw

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens


class FakeClient:
    """Mimics an llm_bridge.LLMClient. `reported_completion` lets a test plant an inflated count."""

    def __init__(self, *, provider="openai", model="gpt-4o", reported_completion=None, raw=None):
        self._provider = provider
        self._model = model
        self._reported_completion = reported_completion
        self._raw = raw
        self.calls = 0

    @property
    def model(self):
        return self._model

    @property
    def provider(self):
        return self._provider

    def chat(self, messages, *, temperature=0.7, max_tokens=1024, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs  # capture what reached the inner client (must exclude tl_*)
        # Honest completion count = real re-tokenised count, unless a test plants an inflation.
        true_out, _ = count_tokens(RESP_TEXT, self._provider, self._model)
        completion = self._reported_completion if self._reported_completion is not None else true_out
        return FakeResponse(RESP_TEXT, self._model, prompt_tokens=20,
                            completion_tokens=completion, raw=self._raw)

    def complete(self, prompt, *, system=None, temperature=0.7, max_tokens=1024, **kwargs):
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        return self.chat(msgs, temperature=temperature, max_tokens=max_tokens, **kwargs)


def _store(tmp="_t_bridge.db"):
    if os.path.exists(tmp):
        os.remove(tmp)
    return Store(tmp), tmp


def test_chat_is_recorded_and_reconciled():
    store, db = _store()
    # Exercise the public metered() factory here (other tests construct the class directly).
    client = metered(FakeClient(), store, session_id="s1")
    resp = client.chat([{"role": "user", "content": "hello"}])
    assert resp.content == RESP_TEXT          # the real response is returned untouched
    recs = store.all_records()
    assert len(recs) == 1
    assert recs[0].session_id == "s1"
    assert recs[0].response_text == RESP_TEXT  # text preserved → output verifiable
    os.remove(db)


def test_complete_is_metered_not_bypassed():
    # RISK #1: complete() must route through the metered chat, not inner.complete.
    store, db = _store()
    inner = FakeClient()
    client = MeteringLLMClient(inner, store, now=_FIXED_TS)
    client.complete("write me a haiku", system="be terse")
    recs = store.all_records()
    assert len(recs) == 1                       # the single-turn call WAS metered
    assert inner.calls == 1                      # and the real call happened exactly once
    assert "be terse" in recs[0].request_text   # system prompt captured for input bounding
    os.remove(db)


def test_output_overcount_caught_end_to_end():
    # RISK #2: plant an inflated completion count; the wrapper must surface an OVERCOUNT
    # (or OUT_OF_BAND in an estimator-only environment) via re-tokenisation of resp.content.
    store, db = _store()
    true_out, conf = count_tokens(RESP_TEXT, "openai", "gpt-4o")
    captured = {}
    client = MeteringLLMClient(
        FakeClient(reported_completion=true_out + 100), store, now=_FIXED_TS,
        on_call=lambda rc: captured.setdefault("rc", rc),
    )
    client.chat([{"role": "user", "content": "hi"}])
    rc = captured["rc"]
    out = next(b for b in rc.buckets if b.bucket == "output")
    assert out.verdict in (Verdict.OVERCOUNT, Verdict.OUT_OF_BAND)
    assert rc.has_overcount
    os.remove(db)


def test_passive_logging_never_breaks_the_call():
    # RISK #3: even if persistence blows up, the caller still gets their response.
    class ExplodingStore:
        redact = False

        def record(self, rec):
            raise RuntimeError("disk full")

    client = MeteringLLMClient(FakeClient(), ExplodingStore(), now=_FIXED_TS)
    resp = client.chat([{"role": "user", "content": "hi"}])
    assert resp.content == RESP_TEXT            # logging failure swallowed; call unaffected


def test_raw_enrichment_is_local_reasoning_and_cache():
    # RISK #4: reasoning/cache come from resp.raw via local adapters, no network, no API call.
    store, db = _store()
    raw = {"usage": {"prompt_tokens": 20, "completion_tokens": 5,
                     "completion_tokens_details": {"reasoning_tokens": 4200},
                     "prompt_tokens_details": {"cached_tokens": 64}}}
    client = MeteringLLMClient(FakeClient(provider="openai", model="o1", raw=raw),
                               store, now=_FIXED_TS)
    client.chat([{"role": "user", "content": "solve"}])
    rec = store.all_records()[0]
    assert rec.reported.reasoning_tokens == 4200   # lifted from raw, not on the normalised resp
    assert rec.reported.cache_read_tokens == 64
    os.remove(db)


def test_per_call_attribution_overrides_and_tags_not_forwarded():
    store, db = _store()
    inner = FakeClient()
    client = MeteringLLMClient(inner, store, user_id="u0", session_id="s0", now=_FIXED_TS)
    client.chat([{"role": "user", "content": "refactor this function and fix the traceback"}],
                tl_user_id="alice", tl_session_id="sess-42", tl_tags={"task_class": "coding"},
                tl_quality=QualitySignal(status="accept", eval_score=0.9))
    rec = store.all_records()[0]
    assert (rec.user_id, rec.session_id) == ("alice", "sess-42")   # per-call override wins
    assert rec.task_class == "coding"                              # explicit tag honoured
    assert rec.quality is not None and rec.quality.status == "accept"  # tl_quality written
    # NON-FORWARDING (#3 PASSIVE): no tl_* kwarg leaks to the inner client.
    for k in inner.last_kwargs:
        assert not k.startswith("tl_"), f"{k} must be stripped before forwarding"
    os.remove(db)


def test_tl_quality_not_forwarded_to_inner_client():
    # The inner client must NEVER receive tl_quality (it is an explicit named param, structurally
    # absent from **kwargs). Also routes through complete() to prove the pass-through path.
    store, db = _store()
    inner = FakeClient()
    client = MeteringLLMClient(inner, store, now=_FIXED_TS)
    client.complete("write a haiku", tl_quality=QualitySignal(status="reject"))
    assert "tl_quality" not in inner.last_kwargs
    assert store.all_records()[0].quality.status == "reject"   # but it WAS metered
    os.remove(db)


def test_metering_client_defines_abc_required_members():
    # When llm_bridge IS installed, MeteringLLMClient subclasses the real LLMClient ABC, whose
    # abstractmethods are exactly {model, chat} (verified against the published base.py). It must
    # define both — plus complete/provider — so the subclass is concrete and instantiable. This
    # pins the ABC code path that the object-fallback fake tests never exercise.
    for attr in ("model", "provider", "chat", "complete"):
        assert attr in MeteringLLMClient.__dict__, f"{attr} must be defined on MeteringLLMClient"


def test_abc_subclass_pattern_is_instantiable_and_complete_dispatches():
    # Faithful local reproduction of llm_bridge's base contract (from base.py): an ABC with
    # abstractmethods {model, chat} and a concrete complete() that dispatches to self.chat.
    # Proves overriding {model, chat} satisfies the ABC (instantiates) and that a single-turn
    # complete() routes through the overridden chat — the pattern MeteringLLMClient relies on.
    import abc

    class _Base(abc.ABC):
        @property
        @abc.abstractmethod
        def model(self):
            ...

        @abc.abstractmethod
        def chat(self, messages, *, temperature=0.7, max_tokens=1024, **kw):
            ...

        def complete(self, prompt, *, system=None, temperature=0.7, max_tokens=1024, **kw):
            msgs = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
            return self.chat(msgs, temperature=temperature, max_tokens=max_tokens, **kw)

    seen = {}

    class _Impl(_Base):
        @property
        def model(self):
            return "m"

        def chat(self, messages, *, temperature=0.7, max_tokens=1024, **kw):
            seen["messages"] = messages
            return "ok"

    impl = _Impl()                              # instantiates → abstractmethods satisfied
    assert impl.complete("hi", system="be terse") == "ok"
    assert seen["messages"][0]["role"] == "system"   # inherited complete reached overridden chat


def test_identity_passthrough():
    store, db = _store()
    client = MeteringLLMClient(FakeClient(provider="bedrock", model="meta-llama-3"), store)
    assert client.provider == "bedrock" and client.model == "meta-llama-3"  # drop-in identity
    os.remove(db)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all llm_bridge connector tests passed")
