# Pivot: from token verifier to agent-accountability over the LiteLLM gateway

Decision record. 2026-07-20.

## The problem with the old product

retoken began as a token *verifier*: independently re-count the tokens a provider bills.
The honesty contract (AGENTS.md) already admitted the fatal shape:

- What retoken can verify EXACTLY — OpenAI / open-weight token counts, via a public tokenizer.
  But nobody doubts these numbers; providers compute them deterministically and return them.
  Re-counting catches ~nothing. It verifies the honest part of the bill.
- What actually costs money and is opaque — reasoning tokens, per-call cache, closed-model
  billing (Claude, Gemini). retoken labels all of it BOUNDED / UNVERIFIABLE. It cannot check the
  thing worth checking.

So the tool verified what didn't need verifying and couldn't verify what did.
The "opacity-first" reframe (commit 1316f55) half-noticed this: it stopped claiming verification
and started measuring how much spend is a black box — honest, but an unactionable number.

**Verification of token counts is dead. Do not revive it.**

## The trigger

The LiteLLM release (July 2026) showed the gateway absorbing the accountability layer:
partial-stream spend on disconnect (#33736), Fireworks cache-read billing (#33714), Gemini
grounding tokens in Usage (#33533), auto prompt-cache injection (#33573). Every one improves the
fidelity of LiteLLM's own spend log. "Reconcile the gateway to catch leaks" is being eaten from
inside — you'd reconcile a number that gets more correct every release.

But the same release made the MCP **toolset grant a real authorization boundary** (#33612, #33586,
#33153, #33584), enforced identically on the MCP and REST paths, acting as a ceiling. LiteLLM is
building the *permission primitives*. It is **not** building the *accountability record* over them.

**Decision: retoken consumes the LiteLLM gateway as its input layer. It stops verifying tokens and
becomes the local, honesty-labelled, per-agent / per-tool spend-and-authorization record the
gateway does not emit.** Reframe around the agent, not the token.

## The product

Local, no-egress artifact = **behaviour ⋈ policy**, read from two LiteLLM Postgres tables (or the
exported StandardLoggingPayload JSONL). Three jobs, all checkable:

1. **Least-privilege audit** — granted toolset vs tools actually called. "Key granted 40 tools,
   used 3 → ceiling 13× too broad." Nobody ships this.
2. **Authorization-conformance** — independently confirm from the logs that no call ever landed
   outside its grant. Verifies the *enforcement* (not the token count). This is what the honesty
   contract was built to express: EXACT where checkable.
3. **Attributed spend** — cost per tool / agent / key. Cross-turn rollups labelled BOUNDED.

## The join — confirmed against LiteLLM source (schema.prisma, types/utils.py, mcp_server/server.py)

### Behaviour — `LiteLLM_SpendLogs`
An MCP `tools/call` (MCP-protocol path OR REST `/mcp`) is metered as its **own spend row** via the
same success-logging pipeline as an LLM call. No separate audit table; the tool call *is* a spend
row. Relevant fields:

- `request_id` (PK), `call_type = 'call_mcp_tool'`, `model = 'MCP: <tool>'`
- `mcp_namespaced_tool_name` (String?) — first-class column, `"<server>/<tool>"`
- `api_key` (hashed virtual key), `user`, `team_id`, `organization_id`, `end_user`, `session_id`,
  `agent_id`
- `spend`, `total_tokens` / `prompt_tokens` / `completion_tokens`, `startTime` / `endTime`
- `metadata` (Json) — carries `mcp_tool_call_metadata` = `StandardLoggingMCPToolCall`:
  `name`, `arguments`, `mcp_server_name`, `namespaced_tool_name`, `mcp_server_cost_info`,
  `mcp_session_id`, `mcp_auth_mode`, `mcp_server_resource`

Isolate MCP rows: `WHERE mcp_namespaced_tool_name IS NOT NULL` (or `call_type = 'call_mcp_tool'`).

Export path (S3/GCS/file): one `StandardLoggingPayload` JSON per call; identity in
`metadata.user_api_key_*`, MCP in `metadata.mcp_tool_call_metadata`. retoken can ingest DB or JSONL.

### Policy — `LiteLLM_ObjectPermissionTable`
Linked to `verification_tokens` (keys), `teams`, `users`, `end_users`, `organizations`. Fields:

- `mcp_toolsets` (String[]) — granted toolset IDs (the release's ceiling)
- `mcp_servers` (String[]) — granted server IDs
- `mcp_access_groups` (String[]) — named server bundles
- `mcp_tool_permissions` (Json) — `{"server_id": ["tool_a","tool_b"]}` per-tool grant
- `blocked_tools` (String[]) — deny list
- `search_tools`, `mcp_tool_search_enabled`

MCP server definitions in `LiteLLM_MCPServerTable`: `server_id`, `server_name`, `allowed_tools`,
`mcp_access_groups`. Resolve a key's effective ceiling from its object_permission grant against
these server definitions, minus `blocked_tools`.

## The two edges — these shape the product, state them honestly

1. **The grant is not on the spend row.** The toolset ceiling is checked at request time and NOT
   persisted per call. retoken must join SpendLogs (behaviour) against ObjectPermissionTable
   (policy) via the key. That join *is* the product; it is the read nobody ships.
2. **`spend` is 0 unless `mcp_server_cost_info` / a cost hook is configured**, and LLM-call →
   triggered-tool-call correlation is not automatic (needs a client-propagated
   `session_id`/`trace_id`). Cross-turn cost rollups are therefore BOUNDED — label them so.

## Access

No hosted "LiteLLM MCP login" — LiteLLM is the self-hosted proxy (Docker + Postgres). Full schema
is now known, so live access is NOT needed to design/build; it is a later validation step to
confirm the MCP fields are non-null in a real deployment. Highest-value input right now: a real
exported spend JSONL containing a few MCP tool-call rows. Otherwise build against synthetic rows
shaped to this schema and validate against a live instance later.

## Status / next

- [x] Verification gate passed: behaviour ⋈ policy join confirmed in source.
- [ ] Get a real MCP spend JSONL (or build synthetic fixtures to this schema).
- [ ] Prototype ingester: read SpendLogs MCP rows + ObjectPermissionTable, emit the three-job
      report (least-privilege, conformance, attributed spend) with EXACT/BOUNDED labels.
- [ ] Rework action.yml around the agent-accountability report.
- [ ] Kill remaining token-verification framing in README/AGENTS.md.
