"""Locks the three ground-truth findings of the MCP accountability audit against
the committed fixtures (see tests/fixtures/mcp/CONTRACT.md). If the maths or the
honesty labels drift, these fail."""

import os

from retoken import mcp_audit

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "mcp")
SPENDLOG = os.path.join(FIX, "litellm_spendlogs.jsonl")
PERMS = os.path.join(FIX, "object_permissions.json")


def _document():
    rows = mcp_audit.load_spend_rows(SPENDLOG)
    policy = mcp_audit.load_permissions(PERMS)
    grants = mcp_audit.resolve_grants(policy)
    reports = mcp_audit.build_reports(rows, grants)
    never_granted = mcp_audit.resolve_server_tools_never_granted(policy, grants)
    return mcp_audit.build_document(reports, never_granted)


def _key(doc, key_hash):
    for k in doc["keys"]:
        if k["key_hash"] == key_hash:
            return k
    raise AssertionError(f"key {key_hash} not in report")


def test_alpha_over_broad_grant():
    """alpha granted 5 tools, used 4; stripe/list_charges is the sole unused grant."""
    lp = _key(_document(), "key-alpha")["least_privilege_audit"]
    assert lp["granted_count"] == 5
    assert lp["used_count"] == 4
    assert lp["unused_grants"] == ["stripe/list_charges"]


def test_beta_conformance_violation():
    """beta called github/create_issue (req-010) outside its grant; exactly one violation overall."""
    viols = _document()["summary"]["violations"]
    assert len(viols) == 1
    v = viols[0]
    assert v["key"] == "key-beta"
    assert v["tool"] == "github/create_issue"
    assert v["request_id"] == "req-010"


def test_alpha_unpriced_is_bounded_not_zero():
    """alpha's stripe/get_charge has no cost info: counted as unpriced/BOUNDED, never a silent 0."""
    mcp = _key(_document(), "key-alpha")["attributed_spend"]["mcp"]
    assert mcp["unpriced_calls"] == 1
    assert mcp["unpriced_label"] == "BOUNDED"


def test_spend_math_matches_fixture():
    """Priced spend is EXACT and matches the actual fixture rows (beta MCP = 0.004, not prose 0.005)."""
    doc = _document()
    alpha = _key(doc, "key-alpha")["attributed_spend"]
    beta = _key(doc, "key-beta")["attributed_spend"]
    assert round(alpha["mcp"]["exact_total"], 6) == 0.009
    assert round(alpha["llm"]["exact_total"], 6) == 0.05
    assert round(beta["mcp"]["exact_total"], 6) == 0.004
    assert round(beta["llm"]["exact_total"], 6) == 0.03


def test_summary_counts():
    """Exactly one of each headline finding."""
    s = _document()["summary"]
    assert s["over_broad_grants"] == 1
    assert s["conformance_violations"] == 1
    assert s["unpriced_spend_flags"] == 1


def test_server_tools_never_granted_not_flagged_as_unused():
    """delete_repo / merge_pr / create_refund were never granted -> informational, not 'unused grants'."""
    doc = _document()
    never = set(doc["server_tools_never_granted"])
    assert {"github/delete_repo", "github/merge_pr", "stripe/create_refund"} <= never
    for k in doc["keys"]:
        assert "github/delete_repo" not in k["least_privilege_audit"]["unused_grants"]
