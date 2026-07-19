# retoken MCP accountability — prototype contract

The prototype ingests a LiteLLM spend export (`litellm_spendlogs.jsonl`, one
`StandardLoggingPayload` JSON per line) plus a policy snapshot (`object_permissions.json`,
mirroring `LiteLLM_ObjectPermissionTable` + `LiteLLM_MCPServerTable`), and emits an
accountability report. No network calls. Behaviour ⋈ policy, joined on the hashed key.

## Inputs
- `litellm_spendlogs.jsonl` — MCP tool-call rows have `call_type == "call_mcp_tool"` and
  `metadata.mcp_tool_call_metadata` non-null (`namespaced_tool_name` = `"<server>/<tool>"`,
  `mcp_server_name`, `mcp_server_cost_info`). LLM rows have `mcp_tool_call_metadata == null`.
  Identity is `metadata.user_api_key_hash` (+ `agent_id`, `team_id`, `session_id`).
- `object_permissions.json` — per-key grant: `mcp_tool_permissions` (`{server: [tools]}`),
  `blocked_tools` (`"server/tool"`), `mcp_toolsets`, plus server `allowed_tools`.

## Required outputs (three jobs)
For each key/agent:

1. **Least-privilege audit** — resolve the effective *granted* tool set (from
   `mcp_tool_permissions` minus `blocked_tools`). Compare to the *used* set (distinct MCP tools
   actually called). Report: granted count, used count, and **unused grants** (granted but never
   called → over-broad ceiling). Label EXACT (both sides are enumerable).

2. **Authorization-conformance** — every MCP call whose tool is NOT in the key's granted set is a
   **violation** (behaviour exceeded policy — enforcement gap or grant drift). List each violation
   (key, agent, tool, request_id). Label EXACT.

3. **Attributed spend** — sum `response_cost` per key/agent, split MCP vs LLM, and per tool. Where
   an MCP tool has no cost info (`mcp_server_cost_info == {}` and `response_cost == 0`), its spend
   is **UNPRICED → BOUNDED**, not zero — count the calls but flag the cost as unknown. LLM and
   priced-MCP spend is EXACT.

Every figure must carry an EXACT / BOUNDED label (honesty contract). Absence of cost is BOUNDED,
never a silent zero.

## Expected findings (ground truth — judges verify against these)
- **key-alpha / billing-agent**: granted 5 tools (github: create_issue, list_issues, get_repo;
  stripe: get_charge, list_charges); used 4 (create_issue, list_issues, get_repo, get_charge);
  **1 unused grant: stripe/list_charges**. Conformance: **0 violations**. Spend: MCP priced =
  3×0.002 + 2×0.001 + 1×0.001 = 0.009 EXACT; stripe/get_charge = 1 call UNPRICED (BOUNDED);
  LLM = 0.05 EXACT. Total accountable = 0.059 EXACT + 1 unpriced MCP call.
- **key-beta / support-agent**: granted 2 tools (github: list_issues, get_repo); used 3
  (list_issues, get_repo, create_issue); **1 violation: github/create_issue (req-010)** — not in
  grant. Spend: MCP = 2×0.001 + 1×0.001 + 1×0.002 = 0.005 EXACT (all priced); LLM = 0.03 EXACT.
- **delete_repo / merge_pr / create_refund**: allowed on the server but never granted to either
  key — must NOT appear as unused *grants* (they were never granted). `github/delete_repo` is
  explicitly blocked for alpha.

A correct report surfaces exactly: 1 over-broad grant (alpha/stripe/list_charges), 1 conformance
violation (beta/github/create_issue), and 1 unpriced-spend flag (alpha/stripe/get_charge).
