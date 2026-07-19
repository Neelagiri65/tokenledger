#!/usr/bin/env python3
"""
retoken mcp-audit — MCP tool-use accountability report (prototype).

Reads a LiteLLM spend export (StandardLoggingPayload JSONL) plus an
object-permission / MCP-server policy snapshot, joins them on the hashed key,
and reports three jobs per key/agent:

  1. Least-privilege audit    — granted vs used MCP tools -> unused grants
                                 (granted but never called = over-broad ceiling).
  2. Authorization conformance — MCP calls outside the granted set = violations
                                 (behaviour exceeded policy: enforcement gap / drift).
  3. Attributed spend          — response_cost per key/agent, MCP vs LLM, per tool,
                                 with an explicit EXACT / BOUNDED honesty split.

Design constraints (non-negotiable):
  - Pure Python 3.11 standard library only. No third-party imports, no network calls.
  - Honesty contract: every figure is labelled EXACT or BOUNDED. Unpriced MCP spend
    (mcp_server_cost_info == {} and response_cost == 0) is BOUNDED / UNPRICED — counted
    but never a silent zero, never folded into an EXACT total.
  - Grant resolution: mcp_tool_permissions minus blocked_tools. A tool merely *allowed
    on a server* but never granted to a key is NOT an unused grant — it is surfaced in a
    separate informational section so the distinction is visible, not hidden.

Standalone usage:
    python mcp_audit.py <litellm_spendlogs.jsonl> <object_permissions.json> [--json]

Architecture: ingest -> resolve-grant -> audit -> render. Each stage is a pure
function over typed dataclasses so the pipeline is testable stage-by-stage.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------
# Domain model
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SpendRow:
    """One StandardLoggingPayload line, normalised to the fields we need."""

    request_id: str
    key_hash: str
    agent_id: Optional[str]
    team_id: Optional[str]
    session_id: Optional[str]
    is_mcp: bool
    server: Optional[str]           # e.g. "github"
    tool: Optional[str]             # e.g. "create_issue"
    namespaced_tool: Optional[str]  # e.g. "github/create_issue"
    response_cost: float
    mcp_cost_info_present: bool     # True if mcp_server_cost_info was a non-empty dict
    status: str

    @property
    def is_unpriced_mcp(self) -> bool:
        """Honesty contract: an MCP call with no cost info and a zero response_cost
        is a genuine unknown, not a free call."""
        return self.is_mcp and (not self.mcp_cost_info_present) and self.response_cost == 0.0


@dataclass(frozen=True)
class Grant:
    """Effective per-key MCP grant: mcp_tool_permissions minus blocked_tools."""

    key_hash: str
    agent_id: Optional[str]
    team_id: Optional[str]
    granted: frozenset  # of "server/tool" namespaced strings


@dataclass
class ConformanceViolation:
    key_hash: str
    agent_id: Optional[str]
    namespaced_tool: str
    request_id: str


@dataclass
class SpendBucket:
    """Spend accumulator with an explicit exact/unpriced split (honesty contract)."""

    exact_total: float = 0.0
    unpriced_calls: int = 0

    def add(self, row: SpendRow) -> None:
        if row.is_unpriced_mcp:
            self.unpriced_calls += 1
        else:
            self.exact_total += row.response_cost

    def to_dict(self) -> dict:
        d = {"exact_total": round(self.exact_total, 6), "exact_total_label": "EXACT"}
        if self.unpriced_calls:
            d["unpriced_calls"] = self.unpriced_calls
            d["unpriced_label"] = "BOUNDED"
        return d


@dataclass
class KeyReport:
    key_hash: str
    agent_id: Optional[str]
    team_id: Optional[str]
    granted_tools: list
    used_tools: list
    unused_grants: list
    violations: list
    llm_spend: SpendBucket = field(default_factory=SpendBucket)
    mcp_spend: SpendBucket = field(default_factory=SpendBucket)
    mcp_spend_by_tool: dict = field(default_factory=dict)  # tool -> SpendBucket

    def to_dict(self) -> dict:
        return {
            "key_hash": self.key_hash,
            "agent_id": self.agent_id,
            "team_id": self.team_id,
            "least_privilege_audit": {
                "label": "EXACT",
                "granted_count": len(self.granted_tools),
                "granted_tools": sorted(self.granted_tools),
                "used_count": len(self.used_tools),
                "used_tools": sorted(self.used_tools),
                "unused_grants": sorted(self.unused_grants),
                "unused_grant_count": len(self.unused_grants),
            },
            "authorization_conformance": {
                "label": "EXACT",
                "violation_count": len(self.violations),
                "violations": [
                    {"tool": v.namespaced_tool, "request_id": v.request_id}
                    for v in self.violations
                ],
            },
            "attributed_spend": {
                "llm": self.llm_spend.to_dict(),
                "mcp": self.mcp_spend.to_dict(),
                "mcp_by_tool": {
                    tool: bucket.to_dict()
                    for tool, bucket in sorted(self.mcp_spend_by_tool.items())
                },
            },
        }


# --------------------------------------------------------------------------
# Stage 1: ingest
# --------------------------------------------------------------------------

def load_spend_rows(path: str) -> list:
    rows: list = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
            rows.append(_row_from_payload(payload))
    return rows


def _row_from_payload(payload: dict) -> SpendRow:
    metadata = payload.get("metadata") or {}
    mcp_meta = metadata.get("mcp_tool_call_metadata")
    is_mcp = payload.get("call_type") == "call_mcp_tool" and mcp_meta is not None

    server = tool = namespaced = None
    mcp_cost_info_present = False
    if is_mcp:
        server = mcp_meta.get("mcp_server_name")
        tool = mcp_meta.get("name")
        namespaced = mcp_meta.get("namespaced_tool_name") or (
            f"{server}/{tool}" if server and tool else None
        )
        cost_info = mcp_meta.get("mcp_server_cost_info")
        mcp_cost_info_present = bool(cost_info)  # non-empty dict -> True

    return SpendRow(
        request_id=payload.get("id", "<unknown>"),
        key_hash=metadata.get("user_api_key_hash", "<unknown>"),
        agent_id=metadata.get("agent_id"),
        team_id=metadata.get("user_api_key_team_id") or metadata.get("team_id"),
        session_id=metadata.get("session_id"),
        is_mcp=is_mcp,
        server=server,
        tool=tool,
        namespaced_tool=namespaced,
        response_cost=float(payload.get("response_cost") or 0.0),
        mcp_cost_info_present=mcp_cost_info_present,
        status=payload.get("status", "unknown"),
    )


def load_permissions(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Stage 2: resolve-grant
# --------------------------------------------------------------------------

def resolve_grants(policy: dict) -> dict:
    """key_hash -> Grant, using mcp_tool_permissions minus blocked_tools.

    Deliberately ignores server-level allowed_tools for grant purposes: a tool
    allowed on a server but never listed in the key's mcp_tool_permissions was
    never granted, and must not surface as an "unused grant".
    """
    grants: dict = {}
    for entry in policy.get("object_permissions", []):
        key_hash = entry["grantee_id"]
        blocked = set(entry.get("blocked_tools") or [])
        granted: set = set()
        for server, tools in (entry.get("mcp_tool_permissions") or {}).items():
            for tool in tools:
                namespaced = f"{server}/{tool}"
                if namespaced in blocked:
                    continue
                granted.add(namespaced)
        grants[key_hash] = Grant(
            key_hash=key_hash,
            agent_id=entry.get("agent_id"),
            team_id=entry.get("team_id"),
            granted=frozenset(granted),
        )
    return grants


def resolve_server_tools_never_granted(policy: dict, grants: dict) -> list:
    """Tools that exist on a server (allowed_tools) but were never granted to ANY key.

    Grafted from the narrative-report strategy: surfacing these explicitly makes the
    honesty distinction visible — they are NOT 'unused grants' (they were never granted),
    they are simply capabilities the server offers that policy never handed out.
    """
    granted_anywhere: set = set()
    for g in grants.values():
        granted_anywhere |= g.granted
    never_granted: set = set()
    for server in policy.get("mcp_servers", []):
        sid = server.get("server_id") or server.get("server_name")
        for tool in server.get("allowed_tools", []) or []:
            ns = f"{sid}/{tool}"
            if ns not in granted_anywhere:
                never_granted.add(ns)
    return sorted(never_granted)


# --------------------------------------------------------------------------
# Stage 3: audit (least-privilege + conformance + attributed spend)
# --------------------------------------------------------------------------

def build_reports(rows: list, grants: dict) -> list:
    rows_by_key: dict = defaultdict(list)
    for row in rows:
        rows_by_key[row.key_hash].append(row)

    all_keys = set(grants.keys()) | set(rows_by_key.keys())
    reports: list = []

    for key_hash in sorted(all_keys):
        grant = grants.get(key_hash)
        granted_set = grant.granted if grant else frozenset()
        key_rows = rows_by_key.get(key_hash, [])

        agent_id = grant.agent_id if grant else None
        team_id = grant.team_id if grant else None
        for row in key_rows:
            agent_id = agent_id or row.agent_id
            team_id = team_id or row.team_id

        used_tools: set = {r.namespaced_tool for r in key_rows if r.is_mcp and r.namespaced_tool}

        unused_grants = sorted(granted_set - used_tools)

        violations = [
            ConformanceViolation(
                key_hash=key_hash,
                agent_id=row.agent_id or agent_id,
                namespaced_tool=row.namespaced_tool,
                request_id=row.request_id,
            )
            for row in key_rows
            if row.is_mcp and row.namespaced_tool and row.namespaced_tool not in granted_set
        ]

        report = KeyReport(
            key_hash=key_hash,
            agent_id=agent_id,
            team_id=team_id,
            granted_tools=sorted(granted_set),
            used_tools=sorted(used_tools),
            unused_grants=unused_grants,
            violations=violations,
        )

        for row in key_rows:
            if row.is_mcp:
                report.mcp_spend.add(row)
                bucket = report.mcp_spend_by_tool.setdefault(
                    row.namespaced_tool or "<unknown>", SpendBucket()
                )
                bucket.add(row)
            else:
                report.llm_spend.add(row)

        reports.append(report)

    return reports


# --------------------------------------------------------------------------
# Stage 4: render
# --------------------------------------------------------------------------

def render_human(reports: list, server_tools_never_granted: list) -> str:
    lines: list = []
    lines.append("=" * 78)
    lines.append("retoken — MCP tool-use accountability report")
    lines.append("=" * 78)
    # Honesty-contract preamble (grafted from the narrative-report strategy): state the
    # meaning of the labels once, up front, so no figure below is ambiguous.
    lines.append("")
    lines.append("Honesty contract: every figure is labelled EXACT (fully enumerable / priced)")
    lines.append("or BOUNDED (a lower bound — unpriced MCP calls exist, are counted separately,")
    lines.append("and are never folded into a total as a silent $0).")

    total_violations = 0
    total_unused = 0
    total_unpriced_calls = 0

    for r in reports:
        lines.append("")
        lines.append(f"Key: {r.key_hash}  (agent={r.agent_id!r}, team={r.team_id!r})")
        lines.append("-" * 78)

        lines.append(
            f"  [1] Least-privilege audit (EXACT): "
            f"granted={len(r.granted_tools)} used={len(r.used_tools)} "
            f"unused_grants={len(r.unused_grants)}"
        )
        for t in r.unused_grants:
            lines.append(f"        UNUSED GRANT: {t}  (granted, never called -> over-broad ceiling)")

        lines.append(f"  [2] Authorization conformance (EXACT): violations={len(r.violations)}")
        for v in r.violations:
            lines.append(
                f"        VIOLATION: tool={v.namespaced_tool} request_id={v.request_id} "
                f"(called but NOT in granted set)"
            )

        lines.append("  [3] Attributed spend:")
        llm = r.llm_spend
        mcp = r.mcp_spend
        lines.append(
            f"        LLM   : {llm.exact_total:.6f} EXACT"
            + (f"  (+{llm.unpriced_calls} unpriced calls BOUNDED)" if llm.unpriced_calls else "")
        )
        lines.append(
            f"        MCP   : {mcp.exact_total:.6f} EXACT"
            + (f"  (+{mcp.unpriced_calls} unpriced calls BOUNDED)" if mcp.unpriced_calls else "")
        )
        for tool, bucket in sorted(r.mcp_spend_by_tool.items()):
            extra = f"  (+{bucket.unpriced_calls} unpriced BOUNDED)" if bucket.unpriced_calls else ""
            lines.append(f"          - {tool}: {bucket.exact_total:.6f} EXACT{extra}")

        accountable_total = llm.exact_total + mcp.exact_total
        lines.append(
            f"        TOTAL ACCOUNTABLE: {accountable_total:.6f} EXACT"
            + (
                f"  + {mcp.unpriced_calls} unpriced MCP call(s) BOUNDED (cost unknown, not zero)"
                if mcp.unpriced_calls
                else ""
            )
        )

        total_violations += len(r.violations)
        total_unused += len(r.unused_grants)
        total_unpriced_calls += mcp.unpriced_calls

    if server_tools_never_granted:
        lines.append("")
        lines.append("-" * 78)
        lines.append("Server tools that exist but were never granted to any key")
        lines.append("(informational — these are NOT 'unused grants', they were never granted):")
        for t in server_tools_never_granted:
            lines.append(f"    - {t}")

    lines.append("")
    lines.append("=" * 78)
    lines.append(
        f"SUMMARY: {total_unused} over-broad grant(s), "
        f"{total_violations} conformance violation(s), "
        f"{total_unpriced_calls} unpriced-spend flag(s)."
    )
    lines.append("=" * 78)
    return "\n".join(lines)


def build_document(reports: list, server_tools_never_granted: list) -> dict:
    """Machine-readable document. The top-level summary roll-up (flat violation list +
    counts) is grafted from the minimal-unix strategy so a platform team can act on the
    aggregate without walking every key."""
    all_violations = [
        {"key": v.key_hash, "agent": v.agent_id, "tool": v.namespaced_tool,
         "request_id": v.request_id}
        for r in reports for v in r.violations
    ]
    return {
        "keys": [r.to_dict() for r in reports],
        "server_tools_never_granted": server_tools_never_granted,
        "summary": {
            "over_broad_grants": sum(len(r.unused_grants) for r in reports),
            "conformance_violations": len(all_violations),
            "violations": all_violations,
            "unpriced_spend_flags": sum(r.mcp_spend.unpriced_calls for r in reports),
        },
    }


def render_json(reports: list, server_tools_never_granted: list) -> str:
    return json.dumps(
        build_document(reports, server_tools_never_granted), indent=2, sort_keys=False
    )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main(argv: list) -> int:
    args = [a for a in argv[1:] if not a.startswith("-")]
    json_only = "--json" in argv[1:]
    if len(args) != 2:
        print(
            f"usage: python {argv[0]} <litellm_spendlogs.jsonl> "
            f"<object_permissions.json> [--json]",
            file=sys.stderr,
        )
        return 2

    spend_path, policy_path = args[0], args[1]

    rows = load_spend_rows(spend_path)
    policy = load_permissions(policy_path)
    grants = resolve_grants(policy)
    reports = build_reports(rows, grants)
    never_granted = resolve_server_tools_never_granted(policy, grants)

    if json_only:
        print(render_json(reports, never_granted))
        return 0

    print(render_human(reports, never_granted))
    print()
    print("--- machine-readable JSON ---")
    print(render_json(reports, never_granted))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
