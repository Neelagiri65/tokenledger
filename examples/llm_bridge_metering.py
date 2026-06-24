"""
Example: inline metering of an llm_bridge client with TokenLedger.

Runs fully OFFLINE — no API keys, no llm_bridge install required — using a tiny duck-typed
client that mimics llm_bridge's LLMResponse contract, with one call deliberately reporting an
inflated completion-token count so you can see the over-count caught live.

In production it is the same code with two lines changed::

    from llm_bridge import create_llm
    from tokenledger.connectors import metered

    llm = metered(create_llm({"provider": "openai", "model": "gpt-4o-mini"}),
                  Store("tokenledger.db"), session_id="prod", on_call=alert)
    llm.complete("...")      # every call logged + reconciled; the rest of your code is unchanged

Run:  .venv/bin/python -m examples.llm_bridge_metering && open tokenledger_bridge_demo.html
"""

from __future__ import annotations

import os

from tokenledger.core import Verdict, count_tokens
from tokenledger.store import Store
from tokenledger.connectors import metered
from tokenledger.dashboard import print_summary, write_html

DB = "tokenledger_bridge_demo.db"
HTML = "tokenledger_bridge_demo.html"

ANSWER = ("Here is a concise refactor of the function with the traceback fixed and a unit "
          "test added. ") * 6


class _Resp:
    def __init__(self, content, model, prompt_tokens, completion_tokens):
        self.content, self.model = content, model
        self.prompt_tokens, self.completion_tokens = prompt_tokens, completion_tokens
        self.latency_ms, self.raw = 42.0, None

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens


class _FakeBridgeClient:
    """Stands in for `create_llm({...})`. `inflate` adds phantom output tokens on the next call."""

    def __init__(self, provider="openai", model="gpt-4o", inflate=0):
        self._provider, self._model, self._inflate = provider, model, inflate

    model = property(lambda self: self._model)
    provider = property(lambda self: self._provider)

    def chat(self, messages, *, temperature=0.7, max_tokens=1024, **kwargs):
        honest, _ = count_tokens(ANSWER, self._provider, self._model)
        return _Resp(ANSWER, self._model, prompt_tokens=30, completion_tokens=honest + self._inflate)

    def complete(self, prompt, *, system=None, temperature=0.7, max_tokens=1024, **kwargs):
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        return self.chat(msgs)


def _alert(rc):
    if rc.has_overcount:
        for b in rc.buckets:
            if b.verdict in (Verdict.OVERCOUNT, Verdict.OUT_OF_BAND):
                print(f"  ⚠ live alert: {rc.record.model} {b.bucket} {b.verdict.value} — {b.note}")


def main() -> None:
    for f in (DB, HTML):
        if os.path.exists(f):
            os.remove(f)
    store = Store(DB)

    print("Metering an honest client (3 calls):")
    honest = metered(_FakeBridgeClient(), store, session_id="prod-eu", on_call=_alert)
    honest.complete("Refactor this function and fix the traceback", system="You are a senior engineer.")
    honest.chat([{"role": "user", "content": "Draft a cold email to follow up with the prospect"}],
                tl_task_class="outreach")
    honest.chat([{"role": "user", "content": "Summarise this email and add to the meeting agenda"}])

    print("Metering a client that over-reports output by 80 tokens (1 call):")
    rogue = metered(_FakeBridgeClient(provider="openai", model="gpt-4o", inflate=80),
                    store, session_id="prod-eu", on_call=_alert)
    rogue.chat([{"role": "user", "content": "Refactor this module"}], tl_task_class="coding")

    print_summary(store)
    print(f"wrote {write_html(store, HTML)}")


if __name__ == "__main__":
    main()
