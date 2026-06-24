# Known limitations: what "verification" means here

TokenLedger re-counts the tokens in the output you received and reconciles them against what the
provider reported. It is worth being precise about what that does and does not prove, because the
naive reading ("we independently verify your bill") overclaims.

## What re-counting actually checks

A provider reports `completion_tokens` for a call. We re-tokenise the **returned text** with the
model's published tokenizer and compare. This is a **consistency check that binds the bill to the
artifact you received** — the same epistemic class as re-totalling a receipt or verifying a checksum.

It is **not** a tautology. The provider's count comes from the tokens the model *generated* (native
to the serving stack); our count is the *canonical encoding of the delivered text*. Those are two
different computation paths, so a match is genuine evidence that generation was canonical and nothing
was dropped or trimmed between generation and delivery.

But it is **weak evidence**, of a narrow thing:

- Against an honest provider it is a near-guaranteed pass.
- It cannot detect overcharging in any bucket you cannot recompute — reasoning, cache, or the rate.
- A rational overcharge would never touch output tokens, the one bucket you *can* recompute. Inflated
  output would also be obvious: the text would look too short for the cost.

So re-counting is an **integrity check**, not independent verification of your true cost.

## A gap is a flag, never a verdict

A nonzero gap has at least three causes, and the number alone cannot distinguish them:

1. The provider over-reported (a billing error, or inflation).
2. The wrong tokenizer (ours differs from the served model's).
3. Legitimate non-canonical generation — decode then re-encode need not preserve the token count.

So TokenLedger never asserts "caught them" on a gap, just as it never asserts "verified" on a match.
A gap is surfaced for investigation. The tokenizer is **pinned by provenance** — the model's official
published tokenizer, recorded with its source and version — and is **never tuned to make a gap
vanish.** Tuning the tokenizer until the bill reconciles is reverse-engineering the provider's config,
not auditing it. Where no authoritative public tokenizer exists (closed models), the bucket is
reported BOUNDED or UNVERIFIABLE, never EXACT.

## What cannot be verified at all

Reasoning and cache tokens are billed but never returned to you. There is no artifact to re-count, so
they are **UNVERIFIABLE** — and on reasoning models they are most of the bill. We record them and label
them honestly; we never assert them.

## Where the real value is

1. **The opacity measurement (the primary output).** Not "your bill checks out," but *how much of your
   bill has any ground truth to check against at all.* Quantifying the unverifiable share
   (reasoning + cache) is the novel, useful signal — it is leverage to push back on a vendor or to
   justify moving a workload to a model that is fully checkable.
2. **Provider-internal consistency (three-way).** Comparing the provider's per-call usage telemetry
   against its own invoice / billing-API total catches the provider's systems disagreeing. Useful and
   sellable — but it compares two *provider* numbers, so it is internal-consistency, not an
   independent oracle. If both are wrong the same way, it catches nothing. Stated as such.
3. **Delivered vs billed.** Catching that you were billed for 1080p / 10s but delivered 480p / 5s.
   This is the same artifact-binding check as the token re-count, applied to media dimensions.
4. **Cost per accepted output.** The token bill is a proxy; utility is the truth. Tying each call's
   cost to an explicit accept/reject signal shows which model is actually cheaper *for your workload*.
   Descriptive today; the migration thesis is being validated separately.

## The honest one-line claim

TokenLedger measures how much of your AI bill has any ground truth to check against, binds the
checkable part to the artifact you received, and surfaces the opaque majority you pay on trust.
