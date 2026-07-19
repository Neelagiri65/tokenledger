# HANDOFF — tokenledger / retoken

Append-only. Newest entry on top. Never consolidate or delete prior entries.

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
