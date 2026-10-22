# HANDOFF — tokenledger / retoken

Append-only. Newest entry on top. Never consolidate or delete prior entries.

---

## 2026-10-23 — LiteLLM example pinned by digest; cost model is missing the Claude 5.5 models
- `docker-compose.yml` commented sidecar example pointed at `ghcr.io/berriai/litellm:main-latest`, the tag that served the malicious LiteLLM 1.107.0 on 2026-10-21. Pinned to `v1.106.2@sha256:7294b90f868c5c38c99b83021c7ba46f9b439e95f8e32964c925c7d2563a03f7` (digest verified against ghcr). Commit bd50a36, pushed.
- **Next:** cost model only knows `claude-opus-4` / `claude-sonnet-4`; add Opus 5.5, Sonnet 5.5, Haiku 5.5 with prices from the claude-api skill (not memory).

## 2026-07-23 — Detour session: no tokenledger work (System-Design Harness Playbook built in vault)

This session ran in the tokenledger directory but did **no tokenledger work** — it distilled the
ByteByteGo Big Archive 2025 PDF (442pp, 213 articles) via a 12-agent Workflow (9 domain distillers
→ synthesiser + evolution analyst + completeness critic, 521k subagent tokens, 0 errors) into a
harness-engineering playbook for real-world system design.

**Output (all in agent-vault, commit 5f14bac, pushed):**
- Playbook: `wiki/concepts/system-design-harness-playbook-2026-07-23.md` — 4 layers: constraint
  harness → 15-law invariant core + decision ordering → volatility-classed tech bindings (a/b/c
  tiers, downhill-only coupling) → evolution harness (trigger + pattern catalogues, quarterly
  binding review) + 9 gap modules + templates (binding record, constraint test, ADR).
- Raw: `raw/research/2026-07-23-bytebytego-system-design-harness/` (12 files).
- MASTER-INDEX pointer added. Not yet kb-ingested for vault-rag — run kb-ingest next vault session.

**Tokenledger NEXT unchanged (from 2026-07-20):** KEEP + fix the 6 legacy reconcile tests (open-weight
engine, not dead weight); fold EXACT recount in as first ingester check; add Nemotron + Seed-Coder
tokenizers.

---

## 2026-07-20 (later still) — Open-weight verification revived + vault indexed

**Decision reversed: do NOT retire the legacy token-reconciliation code.** Prompted by the user
("we also had access to bytedance and nvidia models — can we not use it?"). "Verification is dead"
was too absolute. Live corridor for EXACT recount = **open-weight model (public tokenizer) +
third-party host/reseller biller + per-token pricing** — more teeth than OpenAI-verification because
an open weight is served by many hosts, each with its own chat template + tokenizer version → real,
catchable token-count drift.
- **NVIDIA Nemotron** — fully open, HF tokenizer → EXACT. **ByteDance Seed-Coder** (MIT) → EXACT.
  **ByteDance Doubao Seed 2.x** (Volcano/Ark) → CLOSED → BOUNDED. ("ByteDance access" ≠ verifiable;
  depends on the model.)
- Caveats: consistency check (teeth in mismatches); reasoning tokens BOUNDED even on open models
  unless reasoning text returned; need prompt-as-sent (chat template applied) to recount.
- **Product unifies on one LiteLLM ingester, three checks:** open-weight row → EXACT recount (verify);
  closed row → BOUNDED; MCP tool-call row → behaviour⋈policy accountability.
- Revised NEXT: KEEP + fix the 6 legacy reconcile tests (they're the open-weight engine, not dead
  weight); fold EXACT recount in as the first ingester check; add Nemotron + Seed-Coder tokenizers.

**Vault indexed (kb-ingest).** Saved to agent-vault and available to vault-rag:
- raw/research/2026-07-20-retoken-litellm-agent-accountability.md
- wiki/research/litellm-mcp-agent-accountability.md (new lead article; supersedes llm-token-metering-verification)
- learnings/discoveries/open-weight-verification-is-the-live-niche-2026-07-20.md
- MASTER-INDEX.md + raw/research/INDEX.md updated; token-metering article cross-linked forward.

---

## 2026-07-20 (later) — Two-mode framing + scope-map policy input (backlog)

Reviewed Rick Hightower's article "Securing LiteLLM's MCP Integration" (full 35-page PDF).
Key realisation: it describes LiteLLM **client-library mode**, NOT the **gateway/proxy mode** our
product targets. The author states "LiteLLM isn't a proxy or gateway — it's a client library" and
hand-rolls his own MCP connection + OAuth-2.1 scope enforcement (`scope_mapping` dict +
`_verify_token_scopes` JWT check). No `LiteLLM_SpendLogs`, no `ObjectPermissionTable`, "audit" is
print statements only. So it maps the mode we're NOT in.

Two LiteLLM worlds to be explicit about:
- **Gateway/proxy mode (OUR TARGET)** — native object-permissions/toolsets, `LiteLLM_SpendLogs`
  with per-tool cost, persistent rows. `mcp-audit` reads these directly. No new capture code.
  Enterprise/platform buyers. **Build here first — data is already on disk.**
- **Client-library mode (second front, NOT now)** — no proxy, no spend log, no persistent audit.
  Larger dev population but we'd have to capture the tool calls ourselves. retoken ALREADY has the
  mechanism: `retoken proxy` (wire-level passthrough capture sidecar) could sit in front and feed
  the SAME `mcp-audit` engine. Same audit, different capture front-end.

Backlog additions (not this sprint):
- **B1. Scope-map as an alternate policy input** to `mcp-audit`. Some shops express agent
  permissions as OAuth scopes (tool→scope), not LiteLLM grants. `mcp-audit`'s policy loader could
  accept a scope-map + a per-key scope set and run the identical behaviour⋈policy engine. Different
  policy source, same core.
- **B2. Client-library-mode capture** via the existing `retoken proxy` sidecar → normalise captured
  MCP tool calls into the same row shape `mcp_audit.load_spend_rows` expects. Serves the no-gateway
  population without a new engine.

Decision recorded: **gateway-mode first**; B1/B2 are deferred second fronts.

---

## 2026-07-20 — Strategic pivot + first prototype landed

**What changed (the big one): the product's job was redefined.**
retoken began as a token *verifier* (re-count provider token bills). That job is dead: the one
number you can verify (OpenAI token counts) is the number nobody misreports, and the costly/opaque
numbers (reasoning, cache, closed-model billing) are exactly the ones you can't verify. Decided to
stop verifying tokens and reframe around the **agent**: read the LiteLLM gateway's own logs and
report **what agents did vs what they were allowed to do**, plus attributed spend. LiteLLM is the
*input layer*, not the competitor. Full rationale: `docs/pivot-agent-accountability.md`.

**Verification gate (passed).** Two research subagents confirmed against LiteLLM source that the
join exists locally:
- Behaviour — `LiteLLM_SpendLogs`: MCP `tools/call` is metered as its own row
  (`call_type='call_mcp_tool'`, `mcp_namespaced_tool_name`, `api_key`, `spend`, `session_id`).
- Policy — `LiteLLM_ObjectPermissionTable`: `mcp_toolsets`, `mcp_servers`, `mcp_tool_permissions`
  (`{server:[tools]}`), `blocked_tools`, linked to verification_tokens/teams/users.
- Two edges: (1) the grant is NOT on the spend row — must join behaviour⋈policy on the key;
  (2) spend is 0 unless `mcp_server_cost_info` configured, and cross-turn rollups need a
  client-propagated session_id → BOUNDED.

**Built (design tournament, explicit opt-in).** 4 design-diverse implementations → each run-verified
against committed fixtures → judged → synthesized. Winner: typed-domain-model (37/40), with grafts
from narrative-report (never-granted section, honesty preamble) and minimal-unix (aggregate JSON
roll-up). honesty-native contestant crashed (0). The adversarial verifiers caught a bug in the
CONTRACT (my answer key), not the code: beta MCP spend is 0.004, not the prose's 0.005 (fixture has
one beta list_issues call, not two). Fixed the contract.

**Landed & committed.**
- `2428bb6` — `retoken/mcp_audit.py` (stdlib, standalone), `retoken mcp-audit` CLI subcommand,
  `tests/test_mcp_audit.py` (6 tests lock the 3 findings), CONTRACT.md spend fix.
- `ca22134` — pivot decision record, `tests/fixtures/mcp/` (spend export + policy + contract),
  `action.yml` (composite GitHub Action, previously uncommitted).

Run it: `retoken mcp-audit tests/fixtures/mcp/litellm_spendlogs.jsonl tests/fixtures/mcp/object_permissions.json`
Surfaces: 1 over-broad grant (alpha/stripe/list_charges), 1 conformance violation
(beta/github/create_issue@req-010), 1 unpriced-spend flag (alpha/stripe/get_charge → BOUNDED).

**Known issues / debt.**
- 6 PRE-EXISTING test failures in the legacy suite (`test_reconcile.py` x4, `test_proxy.py`,
  `test_evaluator.py`) — the old token-reconciliation code. Confirmed they fail without today's
  changes too. Untouched; decide whether the legacy verifier code stays or gets retired with the
  pivot.
- Naming: repo/action say "TokenLedger", package is "retoken", AGENTS.md still uses `tokenledger`
  imports. Unresolved.

**NEXT.**
1. Validate against a REAL LiteLLM instance: confirm `mcp_namespaced_tool_name` /
   `mcp_tool_call_metadata` are non-null in practice, and the object-permission grant shape.
   User offered to stand one up / provide a real exported spend JSONL with MCP rows.
2. Add a DB reader (not just JSONL export) — query `LiteLLM_SpendLogs` + `LiteLLM_ObjectPermissionTable`
   directly (read-only, local).
3. Decide fate of the legacy token-verification code + fix or retire its 6 failing tests.
4. Rework README/AGENTS.md around the agent-accountability position; resolve TokenLedger/retoken naming.
5. Point `action.yml` at the new `mcp-audit` report.
