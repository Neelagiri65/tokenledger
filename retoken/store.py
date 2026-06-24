"""
Local SQLite store for call records. Self-hosted, no egress. Content columns hold the
prompt/response text so output tokens can be re-counted; set redact=True to keep only
hashes and token counts when policy forbids storing content at rest.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Iterator

from .core import CallRecord, Usage
from .quality import QualitySignal

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    route TEXT NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    task_class TEXT DEFAULT 'unclassified',
    input_tokens INTEGER, output_tokens INTEGER, reasoning_tokens INTEGER,
    cache_read_tokens INTEGER, cache_creation_tokens INTEGER,
    request_sha TEXT, response_sha TEXT,
    request_text TEXT, response_text TEXT,
    latency_ms REAL,
    quality_score REAL, quality_status TEXT, quality_success INTEGER,
    reported_cost_usd REAL
);
CREATE INDEX IF NOT EXISTS idx_calls_session ON calls(session_id);
CREATE INDEX IF NOT EXISTS idx_calls_user ON calls(user_id);
CREATE INDEX IF NOT EXISTS idx_calls_provider ON calls(provider);
"""

# Columns that may be missing from a DB created by an older TokenLedger. The migration adds any
# that are absent (additive only — never drops/renames). 'task_class' is here too: it was added
# to _SCHEMA in an earlier change without a migration, leaving pre-task_class DBs raising
# OperationalError 'no column named task_class' on record(); this heals that latent gap as well.
_EXPECTED_COLUMNS: dict[str, str] = {
    "task_class": "TEXT DEFAULT 'unclassified'",
    "quality_score": "REAL",
    "quality_status": "TEXT",
    "quality_success": "INTEGER",
    "reported_cost_usd": "REAL",
}


class Store:
    def __init__(self, path: str = "retoken.db", redact: bool = False):
        self.path = path
        self.redact = redact
        self._init()

    def _init(self) -> None:
        with self._conn() as c:
            c.executescript(_SCHEMA)
            self._migrate(c)

    @staticmethod
    def _migrate(c: sqlite3.Connection) -> None:
        """Idempotent, additive-only migration: ALTER TABLE ADD COLUMN for any expected column
        missing from an existing `calls` table. Never drops or renames, so it cannot corrupt data;
        running it on an already-current DB is a no-op."""
        present = {row["name"] for row in c.execute("PRAGMA table_info(calls)").fetchall()}
        for name, decl in _EXPECTED_COLUMNS.items():
            if name not in present:
                c.execute(f"ALTER TABLE calls ADD COLUMN {name} {decl}")

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def record(self, rec: CallRecord) -> int:
        req = "" if self.redact else rec.request_text
        resp = "" if self.redact else rec.response_text
        q_score, q_status, q_success = _quality_cols(rec.quality)
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO calls (ts, provider, model, route, user_id, session_id,
                    task_class, input_tokens, output_tokens, reasoning_tokens, cache_read_tokens,
                    cache_creation_tokens, request_sha, response_sha, request_text,
                    response_text, latency_ms, quality_score, quality_status, quality_success,
                    reported_cost_usd)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (rec.ts, rec.provider, rec.model, rec.route, rec.user_id, rec.session_id,
                 rec.task_class, rec.reported.input_tokens, rec.reported.output_tokens,
                 rec.reported.reasoning_tokens, rec.reported.cache_read_tokens,
                 rec.reported.cache_creation_tokens, rec.request_sha, rec.response_sha,
                 req, resp, rec.latency_ms, q_score, q_status, q_success,
                 rec.reported_cost_usd),
            )
            return int(cur.lastrowid)

    def set_quality(self, call_id: int, signal: QualitySignal) -> bool:
        """Deferred quality update by primary-key id (the value record() returns). Returns True on
        a single row updated, False otherwise. PASSIVE: a write failure or empty signal is
        swallowed and returns False — it never raises into the caller's real work."""
        if signal is None or signal.is_empty():
            return False
        q_score, q_status, q_success = _quality_cols(signal)
        try:
            with self._conn() as c:
                cur = c.execute(
                    """UPDATE calls SET quality_score=?, quality_status=?, quality_success=?
                       WHERE id=?""",
                    (q_score, q_status, q_success, int(call_id)),
                )
                return cur.rowcount == 1
        except Exception:
            return False

    def set_quality_by(self, session_id: str, request_sha: str, signal: QualitySignal) -> int:
        """Deferred quality update by natural key (session_id + request_sha), for callers who did
        not stash the id. Returns the number of rows updated. PASSIVE: a failure or empty signal
        returns 0 and never raises."""
        if signal is None or signal.is_empty():
            return 0
        q_score, q_status, q_success = _quality_cols(signal)
        try:
            with self._conn() as c:
                cur = c.execute(
                    """UPDATE calls SET quality_score=?, quality_status=?, quality_success=?
                       WHERE session_id=? AND request_sha=?""",
                    (q_score, q_status, q_success, session_id, request_sha),
                )
                return int(cur.rowcount)
        except Exception:
            return 0

    def all_records(self) -> list[CallRecord]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM calls ORDER BY id").fetchall()
        return [self._row_to_record(r) for r in rows]

    @staticmethod
    def _row_to_record(r: sqlite3.Row) -> CallRecord:
        return CallRecord(
            provider=r["provider"], model=r["model"], route=r["route"],
            user_id=r["user_id"], session_id=r["session_id"], ts=r["ts"],
            task_class=(r["task_class"] if "task_class" in r.keys() else "unclassified"),
            reported=Usage(
                input_tokens=r["input_tokens"], output_tokens=r["output_tokens"],
                reasoning_tokens=r["reasoning_tokens"], cache_read_tokens=r["cache_read_tokens"],
                cache_creation_tokens=r["cache_creation_tokens"],
            ),
            request_text=r["request_text"] or "", response_text=r["response_text"] or "",
            latency_ms=r["latency_ms"], request_sha=r["request_sha"] or "",
            response_sha=r["response_sha"] or "",
            quality=_quality_from_row(r),
            reported_cost_usd=(r["reported_cost_usd"] if "reported_cost_usd" in r.keys() else None),
        )

    def summary_json(self) -> str:
        return json.dumps([
            {k: v for k, v in dict(rec.__dict__).items() if k not in ("reported", "quality")}
            | {"reported": rec.reported.__dict__,
               "quality": (rec.quality.__dict__ if rec.quality is not None else None)}
            for rec in self.all_records()
        ], indent=2)


def _quality_cols(signal: QualitySignal | None) -> tuple:
    """Map a QualitySignal (or None/empty) to the three column values: score REAL, status TEXT,
    success INTEGER 0/1 (SQLite has no bool). No signal -> all NULL."""
    if signal is None or signal.is_empty():
        return (None, None, None)
    success = None if signal.success is None else (1 if signal.success else 0)
    return (signal.eval_score, signal.status, success)


def _quality_from_row(r: sqlite3.Row) -> QualitySignal | None:
    """Rebuild a QualitySignal from a row, reading each new column with the same `name in r.keys()`
    guard task_class uses, so an older/un-migrated row shape degrades to None (no KeyError). All
    columns NULL -> no signal (None). quality_success 0/1 maps back to bool/None."""
    keys = r.keys()
    score = r["quality_score"] if "quality_score" in keys else None
    status = r["quality_status"] if "quality_status" in keys else None
    raw_success = r["quality_success"] if "quality_success" in keys else None
    success = None if raw_success is None else bool(raw_success)
    if score is None and status is None and success is None:
        return None
    return QualitySignal(eval_score=score, status=status, success=success)
