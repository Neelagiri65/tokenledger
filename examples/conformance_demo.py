"""
Conformance harness demo. Three simulated providers, scored by the invariant battery:
  1. honest      — counts correctly (real tiktoken)            -> all pass
  2. jittery     — same input counts differently each call     -> determinism FAILS (caught)
  3. linear-bias — multiplies every count by 1.10 consistently -> ALL PASS (the honest blind spot)

Run: python -m examples.conformance_demo
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retoken.core import count_tokens          # noqa: E402
from retoken.conformance import run_conformance  # noqa: E402


def _true(text: str) -> int:
    return count_tokens(text, "openai", "gpt-4o")[0]


def honest(text: str) -> int:
    return _true(text)


# jittery: same text returns slightly different counts on successive calls (non-deterministic meter)
_calls = {"n": 0}
def jittery(text: str) -> int:
    _calls["n"] += 1
    return _true(text) + (_calls["n"] % 3)  # 0,1,2,0,1,2... -> identical input, varying count


def linear_bias(text: str) -> int:
    return int(round(_true(text) * 1.10))  # consistent 10% over-count


def main() -> None:
    for name, fn in [("honest", honest), ("jittery", jittery), ("linear-bias (+10%)", linear_bias)]:
        print(f"\n===== provider: {name} =====")
        print(run_conformance(fn))
    print("\nTakeaway: the battery catches instability/non-linearity with no tokenizer — but a "
          "CONSISTENT bias slips through. That gap is closed only by a public tokenizer "
          "(OpenAI/open-weight exact re-count) or a provider-cooperative proof.")


if __name__ == "__main__":
    main()
