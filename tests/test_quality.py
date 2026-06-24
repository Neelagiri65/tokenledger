"""
Tests for the quality-signal layer: capture (record-time + deferred), the idempotent migration
(which also heals the pre-task_class latent gap), and the cost-per-ACCEPTED-output metric.

The metric is built and tested HONESTLY: accept is the sole accepted-count driver, unknown is
NEVER a reject, and a zero denominator renders a literal n/a string (never 0, never a division).
"""

import io
import os
import sqlite3
import sys
import tempfile
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenledger.core import CallRecord, Usage  # noqa: E402
from tokenledger.quality import QualitySignal  # noqa: E402
from tokenledger.store import Store  # noqa: E402
from tokenledger.recorder import record_call  # noqa: E402
from tokenledger.dashboard import (  # noqa: E402
    Rollup, cost_per_accepted, rollup_by, reconcile_all, print_summary, write_html, _CPA_NA,
    _CPA_CAVEAT,
)


def _tmp_db() -> str:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)  # let Store create it fresh
    return path


def _rec(store, *, session_id, sha="", model="gpt-4o", out=100, quality=None):
    rec = CallRecord(
        provider="openai", model=model, route="/chat/completions",
        user_id="u", session_id=session_id, ts="2026-06-20T10:00:00Z",
        reported=Usage(input_tokens=50, output_tokens=out),
        request_text="prompt-" + (sha or session_id), response_text="answer",
        quality=quality,
    )
    return store.record(rec), rec


# --- (a) QualitySignal normalisation / clamp ------------------------------------------

def test_quality_signal_normalisation_and_clamp():
    assert QualitySignal().is_empty()                                   # all None = no signal
    assert QualitySignal(eval_score=1.7).eval_score == 1.0             # clamped, not rejected
    assert QualitySignal(eval_score=-0.3).eval_score == 0.0
    assert QualitySignal(eval_score=0.42).eval_score == 0.42
    assert QualitySignal(status="ACCEPT").status == "accept"           # case/space normalised
    assert QualitySignal(status="garbage").status == "unknown"         # unknown, not raised
    assert QualitySignal(status=None).status is None                   # None stays None
    assert QualitySignal(success=1).success is True
    assert QualitySignal(success=0).success is False
    sig = QualitySignal(status="accept")
    assert sig.is_labeled() and sig.is_accepted() and not sig.is_empty()
    assert QualitySignal(status="reject").is_labeled()
    assert not QualitySignal(status="unknown").is_labeled()            # unknown is NOT labeled
    print("PASS test_quality_signal_normalisation_and_clamp")


# --- (b) migration heals pre-task_class DB AND adds quality columns -------------------

def test_migration_heals_pre_task_class_and_adds_quality_columns():
    path = _tmp_db()
    # Build the verified BROKEN shape: a `calls` table WITHOUT task_class and WITHOUT quality cols,
    # exactly as a pre-task_class TokenLedger left it.
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, provider TEXT, model TEXT, route TEXT, user_id TEXT, session_id TEXT,
            input_tokens INTEGER, output_tokens INTEGER, reasoning_tokens INTEGER,
            cache_read_tokens INTEGER, cache_creation_tokens INTEGER,
            request_sha TEXT, response_sha TEXT, request_text TEXT, response_text TEXT,
            latency_ms REAL
        )"""
    )
    conn.commit()
    # Confirm the pre-existing latent gap: inserting with task_class raises on this old shape.
    raised = False
    try:
        conn.execute("INSERT INTO calls (task_class) VALUES ('coding')")
    except sqlite3.OperationalError as e:
        raised = "no column named task_class" in str(e)
    conn.close()
    assert raised, "expected the verified pre-task_class OperationalError"

    # Store init runs the idempotent migration; record() + set_quality() now succeed.
    store = Store(path)
    cols = {r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(calls)")}
    assert {"task_class", "quality_score", "quality_status", "quality_success"} <= cols
    cid, _ = _rec(store, session_id="s1", quality=QualitySignal(status="accept"))
    assert store.set_quality(cid, QualitySignal(eval_score=0.8, status="accept")) is True
    # Re-init is idempotent (no error, no duplicate columns).
    Store(path)
    os.remove(path)
    print("PASS test_migration_heals_pre_task_class_and_adds_quality_columns")


def test_fresh_db_has_quality_columns():
    path = _tmp_db()
    store = Store(path)
    cid, _ = _rec(store, session_id="s1", quality=QualitySignal(status="accept", eval_score=0.9, success=True))
    rec = store.all_records()[0]
    assert rec.quality.status == "accept" and rec.quality.eval_score == 0.9 and rec.quality.success is True
    os.remove(path)
    print("PASS test_fresh_db_has_quality_columns")


# --- (c) record-time + deferred round-trip -------------------------------------------

def test_record_time_and_deferred_roundtrip():
    path = _tmp_db()
    store = Store(path)
    # record-time via recorder kwarg
    rec = record_call(store, provider="openai", model="gpt-4o", user_id="u", session_id="s-rt",
                      ts="2026-06-20T10:00:00Z", usage=Usage(input_tokens=10, output_tokens=20),
                      request_text="hi", response_text="ok",
                      quality=QualitySignal(status="accept", eval_score=0.7))
    assert store.all_records()[0].quality.status == "accept"

    # deferred by id (lastrowid returned by record())
    cid, _ = _rec(store, session_id="s-id", sha="byid")
    assert store.all_records()[1].quality is None                      # NULL until set
    assert store.set_quality(cid, QualitySignal(status="reject", eval_score=0.2, success=False)) is True
    got = [r for r in store.all_records() if r.session_id == "s-id"][0]
    assert (got.quality.status, got.quality.eval_score, got.quality.success) == ("reject", 0.2, False)

    # deferred by natural key (session_id + request_sha)
    _id, nat = _rec(store, session_id="s-nat", sha="bynat")
    n = store.set_quality_by(nat.session_id, nat.request_sha, QualitySignal(status="accept"))
    assert n == 1
    got2 = [r for r in store.all_records() if r.session_id == "s-nat"][0]
    assert got2.quality.status == "accept"
    os.remove(path)
    print("PASS test_record_time_and_deferred_roundtrip")


def test_empty_signal_is_noop():
    path = _tmp_db()
    store = Store(path)
    cid, rec = _rec(store, session_id="s1")
    assert store.set_quality(cid, QualitySignal()) is False           # empty = no-op
    assert store.set_quality_by(rec.session_id, rec.request_sha, QualitySignal()) == 0
    assert store.all_records()[0].quality is None
    os.remove(path)
    print("PASS test_empty_signal_is_noop")


# --- (d) cost_per_accepted formula ----------------------------------------------------

def test_cost_per_accepted_formula_and_unknown_excluded():
    # numerator = sum billed over labeled {accept,reject}; denom = count(accept).
    # Build a rollup directly so the math is auditable independent of pricing tables.
    ru = Rollup()
    # two accepts ($0.10 each) and one reject ($0.30) -> numerator 0.50, denom 2 -> 0.25
    ru.labeled_calls = 3
    ru.accepted_calls = 2
    ru.labeled_billed_usd = 0.10 + 0.10 + 0.30
    ru.unlabeled_calls = 5                                             # excluded entirely
    cpa = cost_per_accepted(ru)
    assert abs(cpa - 0.25) < 1e-9, cpa                                # rejected cost charged to accepts
    print("PASS test_cost_per_accepted_formula_and_unknown_excluded")


def test_cost_per_accepted_via_rollup_unknown_not_reject():
    # End-to-end through rollup_by: an 'unknown' and a None must NOT count as rejects; they are
    # unlabeled and excluded from the metric. This is the analog of "no text -> UNVERIFIABLE".
    path = _tmp_db()
    store = Store(path)
    _rec(store, session_id="a1", sha="a1", quality=QualitySignal(status="accept"))
    _rec(store, session_id="a2", sha="a2", quality=QualitySignal(status="reject"))
    _rec(store, session_id="a3", sha="a3", quality=QualitySignal(status="unknown"))   # not a reject
    _rec(store, session_id="a4", sha="a4", quality=None)                              # not a reject
    recs = reconcile_all(store)
    ru = rollup_by(recs, "provider")["openai"]
    assert ru.labeled_calls == 2                                      # accept + reject only
    assert ru.accepted_calls == 1
    assert ru.unlabeled_calls == 2                                    # unknown + None
    # cost-per-accepted = billed over the 2 labeled / 1 accept (a real number, not n/a)
    assert isinstance(cost_per_accepted(ru), float)
    os.remove(path)
    print("PASS test_cost_per_accepted_via_rollup_unknown_not_reject")


def test_zero_denominator_renders_na_string():
    # No accepts at all -> literal n/a string, never 0, never a division, never an exception.
    ru = Rollup()
    ru.labeled_calls = 2
    ru.accepted_calls = 0
    ru.labeled_billed_usd = 0.40
    assert cost_per_accepted(ru) == _CPA_NA
    # No labeled calls at all -> same.
    assert cost_per_accepted(Rollup()) == _CPA_NA
    print("PASS test_zero_denominator_renders_na_string")


# --- (e/f) passive failure path -------------------------------------------------------

def test_set_quality_failure_returns_false_without_raising():
    path = _tmp_db()
    store = Store(path)
    # Point the store at a path that cannot be opened so the UPDATE fails inside the guard.
    bad = Store.__new__(Store)
    bad.path = "/nonexistent-dir-xyz/cannot.db"
    bad.redact = False
    # set_quality / set_quality_by must swallow the failure (PASSIVE), never raise.
    assert bad.set_quality(1, QualitySignal(status="accept")) is False
    assert bad.set_quality_by("s", "sha", QualitySignal(status="accept")) == 0
    os.remove(path)
    print("PASS test_set_quality_failure_returns_false_without_raising")


# --- summary_json serialises a quality-bearing row -----------------------------------

def test_summary_json_serialises_quality():
    path = _tmp_db()
    store = Store(path)
    _rec(store, session_id="s1", quality=QualitySignal(status="accept", eval_score=0.5, success=True))
    _rec(store, session_id="s2", quality=None)
    js = store.summary_json()                                         # must not raise on the object
    assert '"status": "accept"' in js
    assert '"quality": null' in js                                    # un-labelled row serialises None
    print("PASS test_summary_json_serialises_quality")


# --- acceptance #5: the section + caveat render in BOTH CLI and HTML ------------------

def test_dashboard_section_and_caveat_render_cli_and_html():
    path = _tmp_db()
    store = Store(path)
    _rec(store, session_id="c1", sha="c1", quality=QualitySignal(status="accept"))
    _rec(store, session_id="c2", sha="c2", quality=QualitySignal(status="reject"))
    _rec(store, session_id="c3", sha="c3", quality=None)

    buf = io.StringIO()
    with redirect_stdout(buf):
        print_summary(store)
    cli = buf.getvalue()
    assert "Cost per accepted output by activity" in cli or "$/accepted" in cli
    assert "accepted" in cli and "labeled" in cli and "unlabeled" in cli
    assert "unlabeled." in cli                                        # the coverage count line
    assert _CPA_CAVEAT in cli                                         # the UNVALIDATED caveat

    html_path = os.path.join(tempfile.gettempdir(), "tl_quality_test.html")
    write_html(store, html_path)
    with open(html_path, encoding="utf-8") as f:
        doc = f.read()
    assert "Cost per accepted output by activity" in doc
    assert "$/accepted" in doc
    assert "UNVALIDATED" in doc and "DESCRIPTIVE" in doc              # caveat present in HTML
    os.remove(html_path)
    os.remove(path)
    print("PASS test_dashboard_section_and_caveat_render_cli_and_html")


def test_na_renders_in_dashboard_when_no_accepts():
    # All rejects -> denominator 0 -> the literal n/a string must appear in CLI + HTML.
    path = _tmp_db()
    store = Store(path)
    _rec(store, session_id="r1", sha="r1", quality=QualitySignal(status="reject"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_summary(store)
    assert _CPA_NA in buf.getvalue()
    html_path = os.path.join(tempfile.gettempdir(), "tl_quality_na.html")
    write_html(store, html_path)
    with open(html_path, encoding="utf-8") as f:
        assert _CPA_NA in f.read()
    os.remove(html_path)
    os.remove(path)
    print("PASS test_na_renders_in_dashboard_when_no_accepts")


def test_row_missing_quality_columns_degrades_to_none():
    # The `name in r.keys()` guard: a row shape that LACKS the quality columns (older/unexpected
    # SELECT) must degrade to quality=None, never raise KeyError. Build such a Row via a
    # column-subset SELECT and feed it straight to the row->signal helper.
    from tokenledger.store import _quality_from_row
    path = _tmp_db()
    store = Store(path)
    _rec(store, session_id="s1", sha="s1", quality=QualitySignal(status="accept"))
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    # SELECT omits quality_score/status/success entirely -> Row has no such keys.
    row = conn.execute("SELECT id, session_id, request_sha FROM calls").fetchone()
    conn.close()
    assert "quality_score" not in row.keys()
    assert _quality_from_row(row) is None  # degrade-to-None, not KeyError
    os.remove(path)
    print("PASS test_row_missing_quality_columns_degrades_to_none")


def test_reported_cost_roundtrips_and_old_db_migrates():
    # The provider's reported $ round-trips through SQLite, and a DB created WITHOUT the column is
    # migrated (additive) on open so reading it degrades to None, never raises.
    path = _tmp_db()
    store = Store(path)
    store.record(CallRecord(provider="openai", model="gpt-4o", route="/v1", user_id="u",
                            session_id="s1", ts="2026-06-24T00:00:00Z",
                            reported=Usage(input_tokens=10, output_tokens=5),
                            reported_cost_usd=0.0123))
    got = store.all_records()[0]
    assert got.reported_cost_usd == 0.0123                  # round-trips

    # simulate an OLD db lacking the column, then let Store migrate it on open
    path2 = _tmp_db()
    conn = sqlite3.connect(path2)
    conn.execute("CREATE TABLE calls (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, provider TEXT, "
                 "model TEXT, route TEXT, user_id TEXT, session_id TEXT, input_tokens INTEGER, "
                 "output_tokens INTEGER, reasoning_tokens INTEGER, cache_read_tokens INTEGER, "
                 "cache_creation_tokens INTEGER, request_sha TEXT, response_sha TEXT, "
                 "request_text TEXT, response_text TEXT, latency_ms REAL)")
    conn.execute("INSERT INTO calls (ts,provider,model,route,user_id,session_id,input_tokens,"
                 "output_tokens) VALUES ('t','openai','gpt-4o','/v1','u','s',1,1)")
    conn.commit(); conn.close()
    store2 = Store(path2)                                   # _init runs the additive migration
    rec2 = store2.all_records()[0]
    assert rec2.reported_cost_usd is None                  # migrated column, no value -> None, no raise
    os.remove(path); os.remove(path2)
    print("PASS test_reported_cost_roundtrips_and_old_db_migrates")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all quality tests passed")
