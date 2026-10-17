"""SQLite telemetry layer (Step 5z).

Single source of truth for "what happened" in FINAQ — graph runs, node
runs, alerts, triage scans, errors. Lives at `data_cache/state.db`.

Why SQLite (not Postgres / external observability):
  - Single-user single-box system. Postgres adds nothing here.
  - File-based — backs up cleanly with the rest of `data_cache/`.
  - No daemon, no port. Crashing FINAQ doesn't leave orphan state.

What we DO record (centrally queryable):
  - Graph-level: run_id, ticker, thesis, status, duration, errors.
  - Node-level: which agent ran, when, how long, did it fail.
  - Alerts (Phase 1 Triage).
  - Triage runs (Phase 1).
  - Standalone error events (centralised log).

Per-LLM-call detail (schema v6 — revises the earlier "LangSmith owns
per-call detail" decision):
  - `llm_calls` records one row per chat-completion call: node, model,
    latency, tokens, cost, and TRUNCATED prompt/response excerpts. This is
    the always-on local baseline so a failed run stays debuggable even when
    LANGSMITH_TRACING was off at run time. **LangSmith** remains the source
    of full, untruncated traces with replay when tracing is enabled; the Run
    Inspector shows both (local excerpts + a LangSmith deep-link).

Run-ID propagation via `contextvars` — the graph wrapper sets
`current_run_id` once per ainvoke, and `_safe_node` reads it without
touching the FinaqState shape (keeps the state clean).

Schema v9 (Mission Control v2, Phase 1 — docs/MISSION_CONTROL_PLAN.md §3):
  - `node_runs` and `llm_calls` are written in TWO phases: a `running` row
    on entry (`start_node_run` / `start_llm_call`), closed on exit
    (`finish_node_run` / `finish_llm_call`), so a live view can see what is
    executing right now. `record_node_run` / `record_llm_call` remain the
    one-shot path for old callers.
  - Full prompts: every message of a call lands in `llm_messages`, content-
    addressed through `prompt_blobs` (sha256) so an unchanged system prompt
    is stored once. `prune_blobs` enforces `PROMPT_RETENTION_DAYS`; the
    clipped excerpts on `llm_calls` stay forever.
  - `spans` records the steps of non-graph flows (CIO cycle: probe → gates →
    decide → budget → drill → summary → notify) via the `span()` context
    manager; `graph_runs.source` says who started a run; `node_runs.pid`
    lets `sweep_stale_runs` tell a crashed process from a slow one.
  - `_connect` sets WAL + busy_timeout so a 1 s reader and two writer
    processes (dashboard, CIO, Telegram) share the file without
    "database is locked".
  - `live_snapshot()` / `run_snapshot()` build the single state document the
    live view and replay render.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import os
import sqlite3
import statistics
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from utils import logger

DB_PATH = Path("data_cache/state.db")
SCHEMA_VERSION = 9  # two-phase node/llm rows, prompt_blobs + llm_messages, spans, source, pid

# Prompt / response excerpts stored per LLM call are clipped to this many
# characters. Full traces live in LangSmith when tracing is on; the local
# excerpt exists so a failed run is debuggable offline.
LLM_EXCERPT_MAX_CHARS = 2000


def _retention_days() -> int:
    try:
        return max(1, int(os.getenv("PROMPT_RETENTION_DAYS", "90")))
    except ValueError:
        return 90


# Full prompts (llm_messages + prompt_blobs) older than this are pruned; set
# PROMPT_RETENTION_DAYS in .env to change the window. Excerpts are kept.
PROMPT_RETENTION_DAYS = _retention_days()

# Error stamped on `running` rows that `sweep_stale_runs` closes.
STALE_ERROR = "stale: process ended without finishing"

# Local-time hours of the scheduled CIO heartbeat; mirrors the
# StartCalendarInterval in deploy/launchd/com.finaq.cio.plist. Only used to
# compute `vitals.next_heartbeat` for the live view.
HEARTBEAT_HOURS_LOCAL = (5, 13)

# Poll cadence the Live tab runs at; echoed in the snapshot so the client can
# size its animations.
LIVE_POLL_S = 1

# Contextvars propagated across asyncio tasks within a single graph invocation.
# Set by `invoke_with_telemetry`; read by `_safe_node` to attach node rows to
# the parent graph run.
current_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_run_id", default=None
)

# Per-node token + cost accumulator. `_safe_node` enters a fresh dict here on
# entry; `utils.openrouter`'s telemetry interceptor reads `response.usage` on
# every LLM call and adds to the dict; `_safe_node` writes the totals to
# node_runs on exit. Default None = "no node active" so direct ad-hoc calls
# (e.g. `agents/qa.py:ask` outside the graph) don't accumulate ghost rows.
node_telemetry_var: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "node_telemetry", default=None
)


def new_node_telemetry(node: str = "") -> dict:
    """Return a fresh accumulator dict for a single node invocation.
    Caller binds it to `node_telemetry_var` for the node's lifetime.
    `node` labels llm_calls rows written while this accumulator is bound
    (empty for callers outside the graph, e.g. the CIO planner)."""
    return {"tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "n_calls": 0, "node": node}


# --- Schema migration -------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS graph_runs (
    run_id      TEXT PRIMARY KEY,
    ticker      TEXT,
    thesis      TEXT,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    duration_s  REAL,
    status      TEXT NOT NULL,                  -- running | completed | failed
    error       TEXT,
    confidence  TEXT,                           -- low | medium | high (from synthesis)
    -- Schema v9 — who started the run: dashboard | cio | telegram | backtest | cli.
    source      TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS node_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT,
    node        TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    duration_s  REAL,
    status      TEXT NOT NULL,                  -- running (v9, entry row) | completed | failed
    error       TEXT,
    -- Step 10c.8 (schema v2) — per-node LLM telemetry. Populated by the
    -- ContextVar accumulator in `utils.openrouter`. Nodes that don't make
    -- LLM calls (load_thesis, monte_carlo) leave these at 0.
    tokens_in   INTEGER NOT NULL DEFAULT 0,
    tokens_out  INTEGER NOT NULL DEFAULT 0,
    cost_usd    REAL    NOT NULL DEFAULT 0.0,
    n_calls     INTEGER NOT NULL DEFAULT 0,
    -- Schema v9 — the process that ran the node, so the stale sweeper can
    -- tell a crashed process from a slow node.
    pid         INTEGER,
    FOREIGN KEY (run_id) REFERENCES graph_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_node_runs_run_id ON node_runs(run_id);
CREATE INDEX IF NOT EXISTS idx_graph_runs_started_at ON graph_runs(started_at);

-- Schema v6 — one row per chat-completion call, written by the telemetry
-- interceptor in utils/openrouter.py whenever a node accumulator is bound.
-- run_id is NULL for calls outside a graph run (CIO planner, ad-hoc Q&A).
-- prompt/response excerpts are clipped to LLM_EXCERPT_MAX_CHARS; full
-- traces stay in LangSmith when tracing is enabled.
CREATE TABLE IF NOT EXISTS llm_calls (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT,
    node              TEXT NOT NULL DEFAULT '',
    ts                TEXT NOT NULL,              -- v9: request start (pre-v9 rows: write time, after the response)
    model             TEXT,
    latency_s         REAL    NOT NULL DEFAULT 0.0,
    tokens_in         INTEGER NOT NULL DEFAULT 0,
    tokens_out        INTEGER NOT NULL DEFAULT 0,
    cost_usd          REAL    NOT NULL DEFAULT 0.0,
    prompt_excerpt    TEXT,
    response_excerpt  TEXT,
    -- Schema v9 — two-phase write: the row is inserted as `running` before
    -- the SDK call and closed afterwards. `attempt` is 1 today (SDK retries
    -- are invisible to the interceptor — see POSTPONED).
    status            TEXT    NOT NULL DEFAULT 'completed',  -- running | completed | failed
    ended_at          TEXT,
    error             TEXT,
    attempt           INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_llm_calls_run_id ON llm_calls(run_id);

-- Schema v9 — full prompts. `prompt_blobs` is content-addressed (sha256 of
-- the message text) so a system prompt repeated on every run is stored once;
-- `llm_messages` lists the messages of one call in order (seq 0 = system,
-- 1 = user, last = the assistant response). `prune_blobs` drops messages of
-- calls older than PROMPT_RETENTION_DAYS and every blob nothing references.
CREATE TABLE IF NOT EXISTS prompt_blobs (
    sha256      TEXT PRIMARY KEY,
    content     TEXT    NOT NULL,
    n_chars     INTEGER NOT NULL,
    first_seen  TEXT    NOT NULL,
    last_seen   TEXT    NOT NULL,
    seen_count  INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS llm_messages (
    call_id   INTEGER NOT NULL REFERENCES llm_calls(id),
    seq       INTEGER NOT NULL,
    role      TEXT    NOT NULL,                  -- system | user | assistant | …
    sha256    TEXT    NOT NULL REFERENCES prompt_blobs(sha256),
    PRIMARY KEY (call_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_llm_messages_sha256 ON llm_messages(sha256);

-- Schema v9 — steps of non-graph flows. The CIO cycle writes one span per
-- step (probe | gates | decide | budget | drill | summary | notify) under
-- parent_kind='cio_run', parent_id=cio_runs.run_id; Discovery / ad-hoc flows
-- can use the same table later. `meta_json` is small (counts, decision,
-- drill_run_id).
CREATE TABLE IF NOT EXISTS spans (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_kind TEXT NOT NULL,
    parent_id   TEXT NOT NULL,
    name        TEXT NOT NULL,
    label       TEXT,                            -- e.g. the ticker
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    status      TEXT NOT NULL,                   -- running | completed | failed
    meta_json   TEXT,
    error       TEXT
);

CREATE INDEX IF NOT EXISTS idx_spans_parent ON spans(parent_kind, parent_id);

CREATE TABLE IF NOT EXISTS alerts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    ticker        TEXT,
    thesis        TEXT,
    severity      INTEGER,
    signal        TEXT,
    evidence_url  TEXT,
    notion_url    TEXT,
    status        TEXT NOT NULL DEFAULT 'pending'  -- pending | acked | dismissed | actioned
);

CREATE TABLE IF NOT EXISTS triage_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    items_scanned   INTEGER,
    alerts_emitted  INTEGER,
    duration_s      REAL,
    error           TEXT
);

CREATE TABLE IF NOT EXISTS errors (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,
    agent     TEXT,
    message   TEXT NOT NULL,
    run_id    TEXT
);

-- Vector-store ingest manifest (schema v7): which filings are in the Pinecone
-- filings index, with how many chunks. Read by has_ticker / the freshness gate.
CREATE TABLE IF NOT EXISTS ingested_filings (
    ticker       TEXT    NOT NULL,
    accession    TEXT    NOT NULL,
    filing_type  TEXT    NOT NULL,
    filed_date   TEXT    NOT NULL DEFAULT '',   -- ISO date from the SGML header; '' if unparsed
    chunks       INTEGER NOT NULL,
    ingested_at  TEXT    NOT NULL,
    PRIMARY KEY (ticker, accession)
);

-- CIO heartbeat / on-demand cycles (Step 11.5).
-- Each `cio_runs` row is one CIO cycle (heartbeat firing or `/cio` invocation).
-- Each `cio_actions` row is one (ticker, thesis) decision the CIO made
-- during that cycle: drill / reuse / dismiss. The dispatcher writes one
-- cio_runs row per cycle; the planner writes one cio_actions row per
-- decision. Mission Control reads both to render "last 20 actions".
CREATE TABLE IF NOT EXISTS cio_runs (
    run_id          TEXT PRIMARY KEY,
    trigger         TEXT NOT NULL,                -- heartbeat | on_demand | catchup
    started_at      TEXT NOT NULL,
    ended_at        TEXT,
    duration_s      REAL,
    status          TEXT NOT NULL,                -- running | completed | failed
    error           TEXT,
    n_actions       INTEGER NOT NULL DEFAULT 0,
    n_drilled       INTEGER NOT NULL DEFAULT 0,
    n_reused        INTEGER NOT NULL DEFAULT 0,
    n_dismissed     INTEGER NOT NULL DEFAULT 0,
    summary         TEXT,                         -- exec summary markdown sent to user
    -- Step 11.19 — model + cost rollup. `model_used` is the MODEL_CIO env at
    -- decide time (one cycle = one model). `total_cost_usd` is the sum of
    -- per-action LLM costs computed from MODEL_PRICING + response.usage.
    model_used      TEXT,
    total_cost_usd  REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS cio_actions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    cio_run_id     TEXT,                        -- parent cio_runs.run_id (NULL for ad-hoc tests)
    ts             TEXT NOT NULL,
    trigger        TEXT NOT NULL,               -- heartbeat | on_demand | catchup
    ticker         TEXT NOT NULL,
    thesis         TEXT,                        -- thesis slug
    action         TEXT NOT NULL,               -- drill | reuse | dismiss
    rationale      TEXT,                        -- short LLM rationale
    drill_run_id   TEXT,                        -- graph_runs.run_id when action=drill
    reuse_run_id   TEXT,                        -- graph_runs.run_id when action=reuse
    confidence     TEXT,                        -- low | medium | high
    decision_json  TEXT,                        -- full CIODecision Pydantic serialised
    -- Step 11.19 — per-call LLM telemetry. Filled by the planner via
    -- node_telemetry_var ContextVar (the openrouter interceptor reads
    -- response.usage and computes USD via MODEL_PRICING). All zero for
    -- gate-shortcut decisions (yo-yo guard) since no LLM call fired.
    model_used     TEXT,                        -- OpenRouter model id used for this call
    tokens_in      INTEGER NOT NULL DEFAULT 0,
    tokens_out     INTEGER NOT NULL DEFAULT 0,
    cost_usd       REAL    NOT NULL DEFAULT 0.0,
    latency_s      REAL    NOT NULL DEFAULT 0.0,
    -- Who produced the decision: llm | gate (yo-yo shortcut) | budget_cap
    -- (demoted drill) | fallback (planner/LLM error). '' on rows written
    -- before this column existed. The yo-yo guard counts ONLY `llm` rows —
    -- counting its own `gate` output locked every pair permanently.
    source         TEXT    NOT NULL DEFAULT '',
    FOREIGN KEY (cio_run_id) REFERENCES cio_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_cio_actions_ts ON cio_actions(ts);
CREATE INDEX IF NOT EXISTS idx_cio_actions_cio_run_id ON cio_actions(cio_run_id);
CREATE INDEX IF NOT EXISTS idx_cio_actions_ticker_thesis ON cio_actions(ticker, thesis, ts);

-- Persistent halo graph (schema v8; Phase 2 Discovery, ARCHITECTURE §13.3).
-- `graph_nodes` = the tickers in a discovered thesis; `graph_edges` = the
-- supplier/customer/peer/competitor links the Discovery agent proposed and
-- then grounded against filings + news. Scoped per thesis slug so re-running
-- Discovery for a slug replaces its rows (delete-by-slug then insert). Read +
-- written by `data/graph.py`; traversal uses recursive CTEs, no graph library.
CREATE TABLE IF NOT EXISTS graph_nodes (
    thesis_slug  TEXT NOT NULL,
    ticker       TEXT NOT NULL,
    name         TEXT NOT NULL DEFAULT '',   -- resolved company name (yfinance cache)
    first_seen   TEXT NOT NULL,              -- ISO ts first discovered
    last_seen    TEXT NOT NULL,              -- ISO ts of the latest discovery run
    PRIMARY KEY (thesis_slug, ticker)
);

CREATE TABLE IF NOT EXISTS graph_edges (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    thesis_slug   TEXT NOT NULL,
    from_ticker   TEXT NOT NULL,
    to_ticker     TEXT NOT NULL,
    type          TEXT NOT NULL,              -- supplier | customer | peer | competitor
    note          TEXT NOT NULL DEFAULT '',
    confidence    REAL NOT NULL DEFAULT 0.0,  -- [0,1] retrieval-grounding score (§13.2)
    grounded      INTEGER NOT NULL DEFAULT 0, -- 1 when confidence >= GROUND_THRESHOLD
    evidence_json TEXT NOT NULL DEFAULT '[]', -- serialised list[Evidence] citations
    as_of         TEXT NOT NULL,              -- ISO ts the edge was grounded
    UNIQUE (thesis_slug, from_ticker, to_ticker)
);

CREATE INDEX IF NOT EXISTS idx_graph_nodes_slug ON graph_nodes(thesis_slug);
CREATE INDEX IF NOT EXISTS idx_graph_edges_slug ON graph_edges(thesis_slug);
CREATE INDEX IF NOT EXISTS idx_graph_edges_from ON graph_edges(thesis_slug, from_ticker);
"""


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    """Open a SQLite connection. Idempotent; creates parent dir if needed."""
    target = Path(db_path) if db_path is not None else DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    # `check_same_thread=False` is safe here because Streamlit's reruns happen
    # in the same process; we don't share connections across threads.
    conn = sqlite3.connect(target, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # Schema v9 — three processes share this file (dashboard, CIO, Telegram)
    # and the Live tab reads it every second. WAL lets readers proceed while a
    # writer commits; busy_timeout makes a contended writer wait instead of
    # raising "database is locked". journal_mode is persistent on the file
    # and a no-op once set; it needs write access, so tolerate a read-only
    # mount.
    conn.execute("PRAGMA busy_timeout = 5000")
    with contextlib.suppress(sqlite3.OperationalError):
        conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(db_path: Path | None = None) -> None:
    """Create the schema if missing + apply forward-only migrations.
    Idempotent — calling twice is a no-op.

    Schema v1 → v2 (Step 10c.8): adds tokens_in / tokens_out / cost_usd /
    n_calls columns to `node_runs`. Existing rows back-fill to 0 (which
    is honest — we didn't track tokens before this migration). New rows
    populate from the ContextVar accumulator.
    """
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA_SQL)
        # Forward-only migrations for DBs created before SCHEMA_VERSION
        # bumps. PRAGMA table_info is the cheapest way to introspect
        # column presence without depending on a dedicated migration tool.
        existing = {
            row["name"] for row in conn.execute("PRAGMA table_info(node_runs)")
        }
        for col_name, col_def in [
            ("tokens_in",  "INTEGER NOT NULL DEFAULT 0"),
            ("tokens_out", "INTEGER NOT NULL DEFAULT 0"),
            ("cost_usd",   "REAL NOT NULL DEFAULT 0.0"),
            ("n_calls",    "INTEGER NOT NULL DEFAULT 0"),
        ]:
            if col_name not in existing:
                conn.execute(
                    f"ALTER TABLE node_runs ADD COLUMN {col_name} {col_def}"
                )

        # Step 11.19 — cio_actions per-call telemetry columns.
        existing_actions = {
            row["name"] for row in conn.execute("PRAGMA table_info(cio_actions)")
        }
        for col_name, col_def in [
            ("model_used", "TEXT"),
            ("tokens_in",  "INTEGER NOT NULL DEFAULT 0"),
            ("tokens_out", "INTEGER NOT NULL DEFAULT 0"),
            ("cost_usd",   "REAL    NOT NULL DEFAULT 0.0"),
            ("latency_s",  "REAL    NOT NULL DEFAULT 0.0"),
            ("source",     "TEXT    NOT NULL DEFAULT ''"),  # schema v5
        ]:
            if col_name not in existing_actions:
                conn.execute(
                    f"ALTER TABLE cio_actions ADD COLUMN {col_name} {col_def}"
                )

        # Step 11.19 — cio_runs cost rollup columns.
        existing_runs = {
            row["name"] for row in conn.execute("PRAGMA table_info(cio_runs)")
        }
        for col_name, col_def in [
            ("model_used",     "TEXT"),
            ("total_cost_usd", "REAL NOT NULL DEFAULT 0.0"),
        ]:
            if col_name not in existing_runs:
                conn.execute(
                    f"ALTER TABLE cio_runs ADD COLUMN {col_name} {col_def}"
                )

        # Schema v9 — Mission Control v2 Phase 1. New tables come from
        # _SCHEMA_SQL (CREATE IF NOT EXISTS); the columns below are added to
        # tables that already existed. Pre-v9 llm_calls rows were written
        # after the response, so they are `completed` (the default) unless
        # the interceptor stamped CALL FAILED, and their `ts` IS the end time.
        if "pid" not in existing:
            conn.execute("ALTER TABLE node_runs ADD COLUMN pid INTEGER")
        existing_graph = {
            row["name"] for row in conn.execute("PRAGMA table_info(graph_runs)")
        }
        if "source" not in existing_graph:
            conn.execute("ALTER TABLE graph_runs ADD COLUMN source TEXT NOT NULL DEFAULT ''")
        existing_calls = {
            row["name"] for row in conn.execute("PRAGMA table_info(llm_calls)")
        }
        for col_name, col_def in [
            ("status",   "TEXT NOT NULL DEFAULT 'completed'"),
            ("ended_at", "TEXT"),
            ("error",    "TEXT"),
            ("attempt",  "INTEGER NOT NULL DEFAULT 1"),
        ]:
            if col_name not in existing_calls:
                conn.execute(f"ALTER TABLE llm_calls ADD COLUMN {col_name} {col_def}")
        if "status" not in existing_calls:
            conn.execute(
                """
                UPDATE llm_calls
                   SET status = 'failed', error = substr(response_excerpt, 14)
                 WHERE response_excerpt LIKE 'CALL FAILED:%'
                """
            )
            conn.execute("UPDATE llm_calls SET ended_at = ts WHERE ended_at IS NULL")

        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        # Retention: prune full prompts at most once a day. Best-effort — a
        # failed prune must never block the write that triggered init_db.
        try:
            _maybe_prune(conn)
        except Exception as exc:
            logger.warning(f"[telemetry] prompt-blob prune skipped: {exc}")


def _maybe_prune(conn: sqlite3.Connection) -> None:
    """Run `_prune_blobs_conn` once per UTC day, remembered in meta.last_prune."""
    today = datetime.now(UTC).date().isoformat()
    row = conn.execute("SELECT value FROM meta WHERE key = 'last_prune'").fetchone()
    if row is not None and str(row["value"]) >= today:
        return
    _prune_blobs_conn(conn, PROMPT_RETENTION_DAYS)
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('last_prune', ?)", (today,)
    )


# --- Time helpers -----------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _new_run_id() -> str:
    return str(uuid.uuid4())


# --- Graph + node recording -------------------------------------------------


def start_graph_run(
    ticker: str,
    thesis: str,
    *,
    source: str = "",
    db_path: Path | None = None,
) -> str:
    """Insert a graph_runs row in the `running` state and return the run_id.
    Caller must call `finish_graph_run` after the run completes. `source`
    (schema v9) names who started it: dashboard | cio | telegram | backtest | cli."""
    init_db(db_path)
    run_id = _new_run_id()
    started_at = _now_iso()
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO graph_runs (run_id, ticker, thesis, started_at, status, source)
            VALUES (?, ?, ?, ?, 'running', ?)
            """,
            (run_id, ticker, thesis, started_at, source or ""),
        )
    return run_id


def finish_graph_run(
    run_id: str,
    status: str,
    *,
    error: str | None = None,
    confidence: str | None = None,
    duration_s: float | None = None,
    db_path: Path | None = None,
) -> None:
    """Update the graph_runs row at completion. `status` ∈ {completed, failed}."""
    if status not in ("completed", "failed"):
        raise ValueError(f"status must be 'completed' or 'failed', got {status!r}")
    ended_at = _now_iso()
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE graph_runs
               SET ended_at = ?, status = ?, error = ?, confidence = ?, duration_s = ?
             WHERE run_id = ?
            """,
            (ended_at, status, error, confidence, duration_s, run_id),
        )


def record_node_run(
    run_id: str | None,
    node: str,
    started_at: str,
    ended_at: str,
    duration_s: float,
    status: str,
    *,
    error: str | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
    n_calls: int = 0,
    db_path: Path | None = None,
) -> int:
    """Insert a node_runs row. `status` ∈ {completed, failed}.

    Token / cost fields default to 0 so legacy callers (and tests) work
    unchanged. `_safe_node` populates them from the ContextVar
    accumulator that `utils/openrouter.py`'s telemetry interceptor
    fills as LLM calls happen inside the node.
    """
    if status not in ("completed", "failed"):
        raise ValueError(f"status must be 'completed' or 'failed', got {status!r}")
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO node_runs (
                run_id, node, started_at, ended_at, duration_s, status, error,
                tokens_in, tokens_out, cost_usd, n_calls
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, node, started_at, ended_at, duration_s, status, error,
                int(tokens_in or 0), int(tokens_out or 0),
                float(cost_usd or 0.0), int(n_calls or 0),
            ),
        )
        return cur.lastrowid or 0


def record_llm_call(
    *,
    run_id: str | None,
    node: str,
    model: str | None,
    latency_s: float,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
    prompt_excerpt: str | None = None,
    response_excerpt: str | None = None,
    db_path: Path | None = None,
) -> int:
    """Append one llm_calls trace row (schema v6). Excerpts are clipped to
    LLM_EXCERPT_MAX_CHARS here so callers can pass raw strings. Written by
    the interceptor in `utils/openrouter.py` on every chat-completion call
    that happens while a node accumulator is bound."""
    init_db(db_path)
    clip = LLM_EXCERPT_MAX_CHARS
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO llm_calls (
                run_id, node, ts, model, latency_s, tokens_in, tokens_out,
                cost_usd, prompt_excerpt, response_excerpt
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, node or "", _now_iso(), model, float(latency_s or 0.0),
                int(tokens_in or 0), int(tokens_out or 0), float(cost_usd or 0.0),
                (prompt_excerpt or "")[:clip] or None,
                (response_excerpt or "")[:clip] or None,
            ),
        )
        return cur.lastrowid or 0


def llm_calls_for_run(run_id: str, *, db_path: Path | None = None) -> list[dict]:
    """Every llm_calls row for a single graph run, in call order. Used by
    the Run Inspector's trace panel.

    Tolerates a pre-v6 DB (no llm_calls table yet): reads must not crash —
    and must not migrate — a database the process has only ever read."""
    if not Path(db_path or DB_PATH).exists():
        return []
    with _connect(db_path) as conn:
        try:
            rows = conn.execute(
                "SELECT * FROM llm_calls WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()
        except sqlite3.OperationalError:
            return []  # schema predates v6; next write migrates it
    return [dict(r) for r in rows]


def record_alert(
    ticker: str,
    thesis: str,
    severity: int,
    signal: str,
    *,
    evidence_url: str | None = None,
    notion_url: str | None = None,
    db_path: Path | None = None,
) -> int:
    """Insert an alert row (Phase 1 Triage). Status starts as 'pending'."""
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO alerts (ts, ticker, thesis, severity, signal, evidence_url, notion_url)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (_now_iso(), ticker, thesis, severity, signal, evidence_url, notion_url),
        )
        return cur.lastrowid or 0


def update_alert_status(alert_id: int, status: str, *, db_path: Path | None = None) -> None:
    if status not in ("pending", "acked", "dismissed", "actioned"):
        raise ValueError(f"unknown alert status {status!r}")
    with _connect(db_path) as conn:
        conn.execute("UPDATE alerts SET status = ? WHERE id = ?", (status, alert_id))


def record_triage_run(
    items_scanned: int,
    alerts_emitted: int,
    duration_s: float,
    *,
    error: str | None = None,
    db_path: Path | None = None,
) -> int:
    """Insert one triage_runs row. Used by Phase 1 scripts/run_triage.py."""
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO triage_runs (ts, items_scanned, alerts_emitted, duration_s, error)
            VALUES (?, ?, ?, ?, ?)
            """,
            (_now_iso(), items_scanned, alerts_emitted, duration_s, error),
        )
        return cur.lastrowid or 0


def record_error(
    agent: str,
    message: str,
    *,
    run_id: str | None = None,
    db_path: Path | None = None,
) -> int:
    """Centralised error log. Agents already log via `utils.logger`; this
    additionally records to state.db so the Mission Control panel can list
    recent errors without grepping log files."""
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO errors (ts, agent, message, run_id)
            VALUES (?, ?, ?, ?)
            """,
            (_now_iso(), agent, message, run_id),
        )
        return cur.lastrowid or 0


# --- Vector-store ingest manifest -------------------------------------------
# Which filings are in the Pinecone filings index, with how many chunks. The
# freshness gate and the "is this ticker ingested?" probes read this instead
# of the index so that dozens of per-ticker checks per CIO cycle stay local.


def record_ingested_filing(
    *,
    ticker: str,
    accession: str,
    filing_type: str,
    filed_date: str,
    chunks: int,
    db_path: Path | None = None,
) -> None:
    """Upsert one filing's manifest row after its vectors landed."""
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO ingested_filings
                (ticker, accession, filing_type, filed_date, chunks, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (ticker.upper(), accession, filing_type, filed_date or "", chunks, _now_iso()),
        )


def clear_ingested_filings(ticker: str, *, db_path: Path | None = None) -> int:
    """Forget every manifest row for `ticker` so the next ingest embeds again.
    Recovery path after the Pinecone index was wiped or renamed."""
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute("DELETE FROM ingested_filings WHERE ticker = ?", (ticker.upper(),))
        return cur.rowcount


def _manifest_conn(db_path: Path | None) -> sqlite3.Connection | None:
    """Read-only entry to the manifest: None when the DB file does not exist
    yet. Connecting would create an empty file, which Mission Control's
    state.db panel reads as "recording" and then fails on missing tables."""
    target = Path(db_path) if db_path is not None else DB_PATH
    return _connect(db_path) if target.exists() else None


def ingested_filings(ticker: str, *, db_path: Path | None = None) -> list[dict]:
    """Manifest rows for `ticker`, newest filed_date first. Read-only: a DB
    created before schema v7 (or not created at all) reads as "nothing
    ingested" — writes migrate it; reads never do, same posture as
    `llm_calls_for_run`."""
    conn = _manifest_conn(db_path)
    if conn is None:
        return []
    try:
        with conn:
            rows = conn.execute(
                "SELECT * FROM ingested_filings WHERE ticker = ? ORDER BY filed_date DESC",
                (ticker.upper(),),
            ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [dict(r) for r in rows]


def ingested_filing_chunks(
    ticker: str, accession: str, *, db_path: Path | None = None
) -> int | None:
    """Chunk count recorded for one (ticker, accession), or None if never ingested."""
    conn = _manifest_conn(db_path)
    if conn is None:
        return None
    try:
        with conn:
            row = conn.execute(
                "SELECT chunks FROM ingested_filings WHERE ticker = ? AND accession = ?",
                (ticker.upper(), accession),
            ).fetchone()
    except sqlite3.OperationalError:
        return None
    return int(row["chunks"]) if row else None


def ingest_manifest_summary(*, db_path: Path | None = None) -> dict:
    """Mission Control card: how much of the filings index the manifest covers."""
    empty = {"tickers": 0, "filings": 0, "chunks": 0, "last_ingested_at": None}
    conn = _manifest_conn(db_path)
    if conn is None:
        return empty
    try:
        with conn:
            row = conn.execute(
                """
                SELECT COUNT(DISTINCT ticker) AS tickers, COUNT(*) AS filings,
                       COALESCE(SUM(chunks), 0) AS chunks, MAX(ingested_at) AS last_ingested_at
                FROM ingested_filings
                """
            ).fetchone()
    except sqlite3.OperationalError:
        return empty
    return {
        "tickers": int(row["tickers"]),
        "filings": int(row["filings"]),
        "chunks": int(row["chunks"]),
        "last_ingested_at": row["last_ingested_at"],
    }


# --- Queries ----------------------------------------------------------------


def recent_runs(limit: int = 20, *, db_path: Path | None = None) -> list[dict]:
    """Most-recent-first list of graph runs, with per-run rollups the runs
    table renders directly: node count, failed-node count + names, summed
    tokens / cost / LLM-call count, and the run-scoped error count (soft
    failures — a run can be 'completed' yet degraded)."""
    if not Path(db_path or DB_PATH).exists():
        return []
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT g.*, COUNT(n.id) AS node_runs_count,
                   SUM(CASE WHEN n.status = 'failed' THEN 1 ELSE 0 END) AS failed_nodes,
                   GROUP_CONCAT(CASE WHEN n.status = 'failed' THEN n.node END) AS failed_node_names,
                   SUM(n.tokens_in)  AS tokens_in,
                   SUM(n.tokens_out) AS tokens_out,
                   SUM(n.cost_usd)   AS cost_usd,
                   SUM(n.n_calls)    AS n_calls,
                   (SELECT COUNT(*) FROM errors e WHERE e.run_id = g.run_id) AS n_errors
              FROM graph_runs g
         LEFT JOIN node_runs n ON n.run_id = g.run_id
          GROUP BY g.run_id
          ORDER BY g.started_at DESC
             LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def recent_node_runs(limit: int = 100, *, db_path: Path | None = None) -> list[dict]:
    if not Path(db_path or DB_PATH).exists():
        return []
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM node_runs ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def node_runs_for_run(run_id: str, *, db_path: Path | None = None) -> list[dict]:
    """Every node_run for a single graph run, ordered by start time.
    Used by the Run Inspector to show the agent-by-agent timeline."""
    if not Path(db_path or DB_PATH).exists():
        return []
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM node_runs WHERE run_id = ? ORDER BY started_at",
            (run_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def node_status_by_run(
    run_ids: list[str], *, db_path: Path | None = None
) -> dict[str, dict[str, str]]:
    """`{run_id: {node: status}}` for many runs in ONE query. Mission
    Control renders per-run agent status dots from this instead of a query
    per row. If a node ran more than once (retry), the latest status wins.
    Empty when the DB is missing or `run_ids` is empty."""
    ids = [r for r in run_ids if r]
    if not ids or not Path(db_path or DB_PATH).exists():
        return {}
    placeholders = ",".join("?" for _ in ids)
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT run_id, node, status
              FROM node_runs
             WHERE run_id IN ({placeholders})
             ORDER BY started_at
            """,
            ids,
        ).fetchall()
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        out.setdefault(r["run_id"], {})[r["node"]] = r["status"]
    return out


def errors_for_run(run_id: str, *, db_path: Path | None = None) -> list[dict]:
    """Error events scoped to a single run_id, ordered by timestamp."""
    if not Path(db_path or DB_PATH).exists():
        return []
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM errors WHERE run_id = ? ORDER BY ts",
            (run_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def daily_cost(days: int = 7, *, db_path: Path | None = None) -> list[dict]:
    """Per-day rollup of cost_usd from node_runs over the last `days` days.
    Used by Mission Control's cost chart and `/status`'s "today's spend".
    Returns rows in chronological order: [{date: 'YYYY-MM-DD', cost_usd: 0.42}, …].
    """
    if not Path(db_path or DB_PATH).exists():
        return []
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                substr(started_at, 1, 10) AS date,
                SUM(cost_usd)             AS cost_usd,
                SUM(tokens_in)            AS tokens_in,
                SUM(tokens_out)           AS tokens_out,
                SUM(n_calls)              AS n_calls
            FROM node_runs
            WHERE started_at >= date('now', ?)
            GROUP BY date
            ORDER BY date
            """,
            (f"-{int(days)} days",),
        ).fetchall()
    return [dict(r) for r in rows]


def cost_today(*, db_path: Path | None = None) -> dict:
    """Aggregate cost + token stats for the UTC date covering 'now'.
    Returns {cost_usd, tokens_in, tokens_out, n_calls}, all 0 when nothing
    happened today."""
    rows = daily_cost(days=1, db_path=db_path)
    today = datetime.now(UTC).date().isoformat()
    for r in rows:
        if r.get("date") == today:
            return {
                "cost_usd":   float(r.get("cost_usd") or 0.0),
                "tokens_in":  int(r.get("tokens_in") or 0),
                "tokens_out": int(r.get("tokens_out") or 0),
                "n_calls":    int(r.get("n_calls") or 0),
            }
    return {"cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0, "n_calls": 0}


def recent_errors(limit: int = 50, *, db_path: Path | None = None) -> list[dict]:
    if not Path(db_path or DB_PATH).exists():
        return []
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM errors ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def recent_alerts(
    limit: int = 50,
    *,
    status: str | None = None,
    db_path: Path | None = None,
) -> list[dict]:
    if not Path(db_path or DB_PATH).exists():
        return []
    sql = "SELECT * FROM alerts"
    args: list[Any] = []
    if status:
        sql += " WHERE status = ?"
        args.append(status)
    sql += " ORDER BY ts DESC LIMIT ?"
    args.append(limit)
    with _connect(db_path) as conn:
        rows = conn.execute(sql, args).fetchall()
    return [dict(r) for r in rows]


def recent_triage_runs(limit: int = 50, *, db_path: Path | None = None) -> list[dict]:
    if not Path(db_path or DB_PATH).exists():
        return []
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM triage_runs ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def daily_run_counts(days: int = 14, *, db_path: Path | None = None) -> list[dict]:
    """Per-day counts of graph_runs (completed + failed). Used by the Mission
    Control daily-runs chart. NOTE: cost-tracking lives in LangSmith for
    Phase 0; if state.db ever needs cost columns, add a small price table
    and compute on insert. For now this is a count-only chart."""
    if not Path(db_path or DB_PATH).exists():
        return []
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT date(started_at) AS day,
                   COUNT(*) AS total,
                   SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed,
                   SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed
              FROM graph_runs
             WHERE started_at >= date('now', '-{int(days)} days')
          GROUP BY day
          ORDER BY day
            """,
        ).fetchall()
    return [dict(r) for r in rows]


def health_summary(*, db_path: Path | None = None) -> dict:
    """Top-of-Mission-Control snapshot: total runs, last run timestamp,
    failure rate over the last 7 days."""
    if not Path(db_path or DB_PATH).exists():
        return {"total_runs": 0, "last_run_at": None, "failure_rate_7d": None}
    with _connect(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM graph_runs").fetchone()["n"]
        last = conn.execute(
            "SELECT started_at FROM graph_runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        last_run_at = last["started_at"] if last else None
        last7 = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed
              FROM graph_runs
             WHERE started_at >= date('now', '-7 days')
            """
        ).fetchone()
        failure_rate = (
            (last7["failed"] or 0) / last7["total"] if last7 and last7["total"] else None
        )
    return {
        "total_runs": total,
        "last_run_at": last_run_at,
        "failure_rate_7d": failure_rate,
    }


# --- Iteration helper for tests --------------------------------------------


def all_node_runs_for(run_id: str, *, db_path: Path | None = None) -> list[dict]:
    if not Path(db_path or DB_PATH).exists():
        return []
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM node_runs WHERE run_id = ? ORDER BY started_at",
            (run_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_graph_run(run_id: str, *, db_path: Path | None = None) -> dict | None:
    if not Path(db_path or DB_PATH).exists():
        return None
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM graph_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    return dict(row) if row else None


# --- CIO cycle + action recording (Step 11.5) -----------------------------


_CIO_TRIGGER_VALUES = ("heartbeat", "on_demand", "catchup")
_CIO_ACTION_VALUES = ("drill", "reuse", "dismiss")
_CIO_STATUS_VALUES = ("running", "completed", "failed")
_CIO_CONFIDENCE_VALUES = ("low", "medium", "high")


def start_cio_run(
    trigger: str,
    *,
    db_path: Path | None = None,
) -> str:
    """Open a new cio_runs row in 'running' state. Returns the run_id.

    Caller must call `finish_cio_run` to finalise (status, counts, summary).
    `trigger` ∈ {heartbeat, on_demand, catchup}.
    """
    if trigger not in _CIO_TRIGGER_VALUES:
        raise ValueError(f"trigger must be one of {_CIO_TRIGGER_VALUES}, got {trigger!r}")
    init_db(db_path)
    run_id = _new_run_id()
    started_at = _now_iso()
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO cio_runs (run_id, trigger, started_at, status)
            VALUES (?, ?, ?, 'running')
            """,
            (run_id, trigger, started_at),
        )
    return run_id


def finish_cio_run(
    run_id: str,
    status: str,
    *,
    error: str | None = None,
    duration_s: float | None = None,
    n_actions: int = 0,
    n_drilled: int = 0,
    n_reused: int = 0,
    n_dismissed: int = 0,
    summary: str | None = None,
    model_used: str | None = None,
    total_cost_usd: float = 0.0,
    db_path: Path | None = None,
) -> None:
    """Close a cio_runs row at completion. `status` ∈ {completed, failed}.

    `model_used` and `total_cost_usd` (Step 11.19) are summed across all
    cio_actions in this cycle by the orchestrator, then passed in here so
    Mission Control can compare per-cycle cost without re-aggregating.
    """
    if status not in ("completed", "failed"):
        raise ValueError(f"status must be 'completed' or 'failed', got {status!r}")
    ended_at = _now_iso()
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE cio_runs
               SET ended_at = ?, status = ?, error = ?, duration_s = ?,
                   n_actions = ?, n_drilled = ?, n_reused = ?, n_dismissed = ?,
                   summary = ?, model_used = ?, total_cost_usd = ?
             WHERE run_id = ?
            """,
            (
                ended_at, status, error, duration_s,
                int(n_actions), int(n_drilled), int(n_reused), int(n_dismissed),
                summary, model_used, float(total_cost_usd or 0.0), run_id,
            ),
        )


def record_cio_action(
    *,
    ticker: str,
    action: str,
    cio_run_id: str | None = None,
    trigger: str = "heartbeat",
    thesis: str | None = None,
    rationale: str | None = None,
    drill_run_id: str | None = None,
    reuse_run_id: str | None = None,
    confidence: str | None = None,
    decision_json: str | None = None,
    model_used: str | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
    latency_s: float = 0.0,
    source: str = "",
    db_path: Path | None = None,
) -> int:
    """Append a cio_actions row. Returns the row id.

    `action` ∈ {drill, reuse, dismiss}. `confidence` ∈ {low, medium, high}.
    `cio_run_id` is the parent cycle id from `start_cio_run` (None ok for
    standalone unit tests).

    Step 11.19 — `model_used`, tokens, cost_usd, and latency_s capture the
    LLM call that produced this decision. All zero/None for gate-shortcut
    decisions (yo-yo guard) since no LLM call fired.

    `source` ∈ {llm, gate, budget_cap, fallback} says who made the call.
    `cio.memory.dismissals_in_window` only counts `llm` rows, so the yo-yo
    guard can never feed on its own shortcuts.
    """
    if action not in _CIO_ACTION_VALUES:
        raise ValueError(f"action must be one of {_CIO_ACTION_VALUES}, got {action!r}")
    if trigger not in _CIO_TRIGGER_VALUES:
        raise ValueError(f"trigger must be one of {_CIO_TRIGGER_VALUES}, got {trigger!r}")
    if confidence is not None and confidence not in _CIO_CONFIDENCE_VALUES:
        raise ValueError(
            f"confidence must be one of {_CIO_CONFIDENCE_VALUES} or None, got {confidence!r}"
        )
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO cio_actions (
                cio_run_id, ts, trigger, ticker, thesis, action, rationale,
                drill_run_id, reuse_run_id, confidence, decision_json,
                model_used, tokens_in, tokens_out, cost_usd, latency_s, source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cio_run_id, _now_iso(), trigger, ticker, thesis, action, rationale,
                drill_run_id, reuse_run_id, confidence, decision_json,
                model_used,
                int(tokens_in or 0), int(tokens_out or 0),
                float(cost_usd or 0.0), float(latency_s or 0.0),
                source or "",
            ),
        )
        return cur.lastrowid or 0


def recent_cio_runs(limit: int = 20, *, db_path: Path | None = None) -> list[dict]:
    """Most-recent-first list of CIO cycles (heartbeat / on-demand / catchup)."""
    if not Path(db_path or DB_PATH).exists():
        return []
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM cio_runs ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def recent_cio_actions(
    limit: int = 20,
    *,
    ticker: str | None = None,
    thesis: str | None = None,
    cio_run_id: str | None = None,
    db_path: Path | None = None,
) -> list[dict]:
    """Most-recent-first list of CIO decisions. Optionally scope to a
    specific (ticker, thesis) pair — used by the planner's cooldown gate —
    or to one parent cycle via `cio_run_id` (Run Inspector's cycle panel;
    filtering in SQL instead of over a recency window means old cycles
    still show their actions)."""
    if not Path(db_path or DB_PATH).exists():
        return []
    sql = "SELECT * FROM cio_actions"
    args: list[Any] = []
    conds: list[str] = []
    if ticker is not None:
        conds.append("ticker = ?")
        args.append(ticker)
    if thesis is not None:
        conds.append("thesis = ?")
        args.append(thesis)
    if cio_run_id is not None:
        conds.append("cio_run_id = ?")
        args.append(cio_run_id)
    if conds:
        sql += " WHERE " + " AND ".join(conds)
    sql += " ORDER BY ts DESC LIMIT ?"
    args.append(limit)
    with _connect(db_path) as conn:
        rows = conn.execute(sql, args).fetchall()
    return [dict(r) for r in rows]


def cio_action_for_run(run_id: str, *, db_path: Path | None = None) -> dict | None:
    """The CIO decision that triggered (or reused) a graph run, or None for
    manual runs. Joins the parent cycle so the Run Inspector can render
    "triggered by CIO heartbeat <ts> — decision: drill (confidence high)"."""
    if not run_id or not Path(db_path or DB_PATH).exists():
        return None
    with _connect(db_path) as conn:
        try:
            row = conn.execute(
                """
                SELECT a.*, r.started_at AS cycle_started_at, r.trigger AS cycle_trigger
                  FROM cio_actions a
             LEFT JOIN cio_runs r ON r.run_id = a.cio_run_id
                 WHERE a.drill_run_id = ? OR a.reuse_run_id = ?
              ORDER BY a.ts DESC LIMIT 1
                """,
                (run_id, run_id),
            ).fetchone()
        except sqlite3.OperationalError:
            return None  # pre-CIO schema — treat as a manual run
    return dict(row) if row else None


def last_drill_for(
    ticker: str,
    thesis: str | None = None,
    *,
    db_path: Path | None = None,
) -> dict | None:
    """Most-recent COMPLETED graph_run for (ticker, thesis), or None.

    The CIO planner uses this to evaluate the cooldown gate: if the most
    recent drill for the pair landed within the cooldown window, the
    planner prefers reuse / dismiss over a fresh drill.

    Note: looks at `graph_runs` regardless of trigger — a manual user
    drill counts toward the cooldown too, since fresh evidence is fresh
    evidence whoever produced it.
    """
    if not Path(db_path or DB_PATH).exists():
        return None
    sql = """
        SELECT * FROM graph_runs
        WHERE ticker = ? AND status = 'completed'
        """
    args: list[Any] = [ticker]
    if thesis is not None:
        sql += " AND thesis = ?"
        args.append(thesis)
    sql += " ORDER BY started_at DESC LIMIT 1"
    with _connect(db_path) as conn:
        row = conn.execute(sql, args).fetchone()
    return dict(row) if row else None


def cio_model_performance(
    *,
    days: int = 30,
    db_path: Path | None = None,
) -> list[dict]:
    """Rollup of CIO per-call metrics grouped by `model_used`, last `days` days.

    Used by Mission Control's "CIO model performance" panel so the user
    can see which model is producing the best decisions, fastest, cheapest.
    A `parse_fail` is detected by rationale starting with "LLM response
    unparseable" (the deterministic-fallback case in
    `cio.planner.decide`).

    Returns rows like:
      {
        "model_used": "openai/gpt-5.4-mini",
        "n_calls":    154,
        "n_drills":   8,
        "n_reuses":   3,
        "n_dismisses": 143,
        "n_parse_fails": 11,
        "avg_latency_s": 4.2,
        "total_cost_usd": 0.34,
        "avg_tokens_in":  1840,
        "avg_tokens_out": 230,
      }
    """
    if not Path(db_path or DB_PATH).exists():
        return []
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                COALESCE(model_used, '(none — gate shortcut)') AS model_used,
                COUNT(*)                                          AS n_calls,
                SUM(CASE WHEN action='drill'    THEN 1 ELSE 0 END) AS n_drills,
                SUM(CASE WHEN action='reuse'    THEN 1 ELSE 0 END) AS n_reuses,
                SUM(CASE WHEN action='dismiss'  THEN 1 ELSE 0 END) AS n_dismisses,
                SUM(CASE WHEN rationale LIKE 'LLM response unparseable%'
                          OR rationale LIKE 'LLM call failed%'
                         THEN 1 ELSE 0 END)                       AS n_parse_fails,
                AVG(latency_s)                                    AS avg_latency_s,
                SUM(cost_usd)                                     AS total_cost_usd,
                AVG(tokens_in)                                    AS avg_tokens_in,
                AVG(tokens_out)                                   AS avg_tokens_out
              FROM cio_actions
             WHERE ts >= datetime('now', '-{int(days)} days')
          GROUP BY model_used
          ORDER BY n_calls DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def last_successful_cio_run_at(*, db_path: Path | None = None) -> str | None:
    """ISO timestamp of the most-recent COMPLETED CIO cycle, or None.

    Used by the dispatcher's catch-up gate at boot (RunAtLoad=true): if
    `now - last_successful > 8h`, fire one catchup cycle so a missed
    heartbeat (Mac was off / droplet was rebooted) doesn't go unnoticed.
    """
    if not Path(db_path or DB_PATH).exists():
        return None
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT ended_at FROM cio_runs
            WHERE status = 'completed'
            ORDER BY ended_at DESC LIMIT 1
            """
        ).fetchone()
    if row is None or row["ended_at"] is None:
        return None
    return str(row["ended_at"])


# --- Schema v9: two-phase node rows (Mission Control v2, Phase 1) -----------


def start_node_run(
    run_id: str | None,
    node: str,
    *,
    started_at: str | None = None,
    pid: int | None = None,
    db_path: Path | None = None,
) -> int:
    """Insert a `node_runs` row in the `running` state at node ENTRY and return
    its id; `finish_node_run` closes it at exit. `pid` (default: this process)
    lets `sweep_stale_runs` tell a crashed process from a slow node.
    `record_node_run` remains the one-shot path for old callers."""
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO node_runs (run_id, node, started_at, status, pid)
            VALUES (?, ?, ?, 'running', ?)
            """,
            (run_id, node, started_at or _now_iso(), os.getpid() if pid is None else int(pid)),
        )
        return cur.lastrowid or 0


def finish_node_run(
    node_run_id: int,
    status: str,
    *,
    ended_at: str | None = None,
    duration_s: float | None = None,
    error: str | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
    n_calls: int = 0,
    db_path: Path | None = None,
) -> None:
    """Close a `node_runs` row opened by `start_node_run`. `status` ∈ {completed, failed}."""
    if status not in ("completed", "failed"):
        raise ValueError(f"status must be 'completed' or 'failed', got {status!r}")
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE node_runs
               SET ended_at = ?, duration_s = ?, status = ?, error = ?,
                   tokens_in = ?, tokens_out = ?, cost_usd = ?, n_calls = ?
             WHERE id = ?
            """,
            (
                ended_at or _now_iso(),
                duration_s,
                status,
                error,
                int(tokens_in or 0),
                int(tokens_out or 0),
                float(cost_usd or 0.0),
                int(n_calls or 0),
                int(node_run_id),
            ),
        )


# --- Schema v9: two-phase LLM calls + full prompts --------------------------


def _message_parts(message: Any) -> tuple[str, str]:
    """(role, text) for one chat message — dicts or SDK objects. Non-string
    content (multi-part messages) is JSON-serialised so it still hashes."""
    if isinstance(message, dict):
        role, content = message.get("role"), message.get("content")
    else:
        role, content = getattr(message, "role", None), getattr(message, "content", None)
    if content is None:
        text = ""
    elif isinstance(content, str):
        text = content
    else:
        text = json.dumps(content, default=str, ensure_ascii=False)
    return str(role or "unknown"), text


def _store_messages(
    conn: sqlite3.Connection, call_id: int, messages: list[Any], *, start_seq: int
) -> int:
    """Insert `messages` as llm_messages rows (seq from `start_seq`), content-
    addressed through prompt_blobs: an unchanged system prompt is stored once
    and its seen_count / last_seen are bumped. Returns rows written."""
    now = _now_iso()
    for offset, message in enumerate(messages):
        role, text = _message_parts(message)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        conn.execute(
            """
            INSERT INTO prompt_blobs (sha256, content, n_chars, first_seen, last_seen, seen_count)
            VALUES (?, ?, ?, ?, ?, 1)
            ON CONFLICT(sha256) DO UPDATE
               SET seen_count = seen_count + 1, last_seen = excluded.last_seen
            """,
            (digest, text, len(text), now, now),
        )
        conn.execute(
            "INSERT OR REPLACE INTO llm_messages (call_id, seq, role, sha256) VALUES (?, ?, ?, ?)",
            (call_id, start_seq + offset, role, digest),
        )
    return len(messages)


def _next_seq(conn: sqlite3.Connection, call_id: int) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(seq), -1) + 1 AS next FROM llm_messages WHERE call_id = ?",
        (call_id,),
    ).fetchone()
    return int(row["next"])


def start_llm_call(
    *,
    run_id: str | None,
    node: str,
    model: str | None,
    messages: list[Any] | None = None,
    prompt_excerpt: str | None = None,
    attempt: int = 1,
    db_path: Path | None = None,
) -> int:
    """Phase 1 of the two-phase `llm_calls` write: insert the row as `running`
    BEFORE the SDK call, with the full request messages stored through
    prompt_blobs in the same transaction. Returns the call id that
    `finish_llm_call` closes. `prompt_excerpt` keeps the pre-v9 readers fed."""
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO llm_calls (run_id, node, ts, model, status, attempt, prompt_excerpt)
            VALUES (?, ?, ?, ?, 'running', ?, ?)
            """,
            (
                run_id,
                node or "",
                _now_iso(),
                model,
                int(attempt or 1),
                (prompt_excerpt or "")[:LLM_EXCERPT_MAX_CHARS] or None,
            ),
        )
        call_id = cur.lastrowid or 0
        if messages:
            _store_messages(conn, call_id, list(messages), start_seq=0)
        return call_id


def finish_llm_call(
    call_id: int,
    *,
    status: str,
    latency_s: float,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
    response_excerpt: str | None = None,
    response_text: str | None = None,
    error: str | None = None,
    db_path: Path | None = None,
) -> None:
    """Phase 2: close the row with the outcome. `status` ∈ {completed, failed}.
    `response_text` (the full assistant message) is appended to llm_messages;
    `response_excerpt` is clipped and kept for the pre-v9 readers."""
    if status not in ("completed", "failed"):
        raise ValueError(f"status must be 'completed' or 'failed', got {status!r}")
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE llm_calls
               SET status = ?, ended_at = ?, latency_s = ?, tokens_in = ?, tokens_out = ?,
                   cost_usd = ?, response_excerpt = ?, error = ?
             WHERE id = ?
            """,
            (
                status,
                _now_iso(),
                float(latency_s or 0.0),
                int(tokens_in or 0),
                int(tokens_out or 0),
                float(cost_usd or 0.0),
                (response_excerpt or "")[:LLM_EXCERPT_MAX_CHARS] or None,
                (error or "")[:1000] or None,
                int(call_id),
            ),
        )
        if response_text is not None:
            _store_messages(
                conn,
                call_id,
                [{"role": "assistant", "content": response_text}],
                start_seq=_next_seq(conn, call_id),
            )


def record_llm_messages(
    call_id: int,
    messages: list[Any],
    *,
    response: str | None = None,
    db_path: Path | None = None,
) -> int:
    """Append `messages` (and, when given, the assistant `response`) to an
    existing call after the rows it already has, deduplicated through
    prompt_blobs. Returns rows written. The interceptor stores messages via
    `start_llm_call` / `finish_llm_call`; this is for callers that only hold a
    call id (tests, backfills, one-shot `record_llm_call` users)."""
    init_db(db_path)
    items = list(messages or [])
    if response is not None:
        items.append({"role": "assistant", "content": response})
    if not items:
        return 0
    with _connect(db_path) as conn:
        return _store_messages(conn, call_id, items, start_seq=_next_seq(conn, call_id))


def llm_messages_for_call(call_id: int, *, db_path: Path | None = None) -> list[dict]:
    """Full messages of one call in order: [{seq, role, text, sha256, n_chars}].
    Empty for a missing file or a pre-v9 DB (reads never migrate)."""
    if not Path(db_path or DB_PATH).exists():
        return []
    with _connect(db_path) as conn:
        try:
            rows = conn.execute(
                """
                SELECT m.seq, m.role, b.content AS text, m.sha256, b.n_chars
                  FROM llm_messages m
                  JOIN prompt_blobs b ON b.sha256 = m.sha256
                 WHERE m.call_id = ?
                 ORDER BY m.seq
                """,
                (int(call_id),),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [dict(r) for r in rows]


def _prune_blobs_conn(conn: sqlite3.Connection, days: int) -> dict:
    cutoff = (datetime.now(UTC) - timedelta(days=int(days))).isoformat()
    messages = conn.execute(
        "DELETE FROM llm_messages WHERE call_id IN (SELECT id FROM llm_calls WHERE ts < ?)",
        (cutoff,),
    ).rowcount
    blobs = conn.execute(
        "DELETE FROM prompt_blobs WHERE sha256 NOT IN (SELECT sha256 FROM llm_messages)"
    ).rowcount
    return {"messages": int(messages), "blobs": int(blobs)}


def prune_blobs(days: int | None = None, *, db_path: Path | None = None) -> dict:
    """Drop the llm_messages rows of calls older than `days` (default
    PROMPT_RETENTION_DAYS) and every prompt_blob nothing references any more.
    A blob shared with a newer call survives. The llm_calls rows and their
    excerpts stay forever. `init_db` runs this once a day (meta.last_prune)."""
    init_db(db_path)
    with _connect(db_path) as conn:
        return _prune_blobs_conn(conn, PROMPT_RETENTION_DAYS if days is None else days)


# --- Schema v9: spans (steps of non-graph flows) -----------------------------


@dataclass
class Span:
    """Handle yielded by `span()`. Mutate `meta` inside the block (it is
    written at exit) and set `error` to record a soft failure without raising."""

    id: int | None
    parent_kind: str
    parent_id: str | None
    name: str
    label: str | None = None
    meta: dict = field(default_factory=dict)
    error: str | None = None


def start_span(
    parent_kind: str,
    parent_id: str,
    name: str,
    *,
    label: str | None = None,
    meta: dict | None = None,
    db_path: Path | None = None,
) -> int:
    """Insert a `running` spans row and return its id."""
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO spans (parent_kind, parent_id, name, label, started_at, status, meta_json)
            VALUES (?, ?, ?, ?, ?, 'running', ?)
            """,
            (
                parent_kind,
                parent_id,
                name,
                label,
                _now_iso(),
                json.dumps(meta, default=str) if meta else None,
            ),
        )
        return cur.lastrowid or 0


def finish_span(
    span_id: int,
    status: str,
    *,
    meta: dict | None = None,
    error: str | None = None,
    db_path: Path | None = None,
) -> None:
    """Close a spans row. `status` ∈ {completed, failed}."""
    if status not in ("completed", "failed"):
        raise ValueError(f"status must be 'completed' or 'failed', got {status!r}")
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE spans SET ended_at = ?, status = ?, meta_json = ?, error = ? WHERE id = ?",
            (
                _now_iso(),
                status,
                json.dumps(meta, default=str) if meta else None,
                (error or "")[:1000] or None,
                int(span_id),
            ),
        )


def _close_span(handle: Span, status: str, *, error: str | None, db_path: Path | None) -> None:
    if handle.id is None:
        return
    try:
        finish_span(handle.id, status, meta=handle.meta, error=error, db_path=db_path)
    except Exception as exc:
        logger.warning(f"[telemetry] span close failed for {handle.name}: {exc}")


@contextlib.contextmanager
def span(
    parent_kind: str,
    parent_id: str | None,
    name: str,
    *,
    label: str | None = None,
    meta: dict | None = None,
    db_path: Path | None = None,
) -> Iterator[Span]:
    """Record one step of a non-graph flow (the CIO cycle today) as a `spans`
    row: `running` on entry, `completed` or `failed` on exit; an exception
    inside the block is recorded and re-raised. The bookkeeping itself is
    best-effort — a telemetry failure is logged, never raised — and with
    `parent_id=None` nothing is written, so call sites need no branch."""
    handle = Span(
        id=None,
        parent_kind=parent_kind,
        parent_id=parent_id,
        name=name,
        label=label,
        meta=dict(meta or {}),
    )
    if parent_id:
        try:
            handle.id = start_span(
                parent_kind, parent_id, name, label=label, meta=handle.meta, db_path=db_path
            )
        except Exception as exc:
            logger.warning(f"[telemetry] span start failed for {name}: {exc}")
    try:
        yield handle
    except Exception as exc:
        _close_span(handle, "failed", error=str(exc), db_path=db_path)
        raise
    _close_span(
        handle, "failed" if handle.error else "completed", error=handle.error, db_path=db_path
    )


def _span_dict(row: sqlite3.Row) -> dict:
    out = dict(row)
    raw = out.pop("meta_json", None)
    try:
        out["meta"] = json.loads(raw) if raw else {}
    except ValueError:
        out["meta"] = {}
    return out


def spans_for(parent_kind: str, parent_id: str, *, db_path: Path | None = None) -> list[dict]:
    """Every span of one parent in start order, with `meta` parsed from meta_json."""
    if not Path(db_path or DB_PATH).exists():
        return []
    with _connect(db_path) as conn:
        try:
            rows = conn.execute(
                "SELECT * FROM spans WHERE parent_kind = ? AND parent_id = ? ORDER BY id",
                (parent_kind, parent_id),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [_span_dict(r) for r in rows]


# --- Schema v9: stale sweeper + medians --------------------------------------


def _pid_alive(pid: int | None) -> bool:
    """True when a process with this pid exists (signal 0 probe)."""
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    except (OverflowError, ValueError, OSError):
        return False
    return True


def _mark_stale(
    conn: sqlite3.Connection, table: str, key_col: str, keys: list[Any], now: str
) -> int:
    if not keys:
        return 0
    placeholders = ",".join("?" for _ in keys)
    cur = conn.execute(
        f"""
        UPDATE {table}
           SET status = 'failed', ended_at = ?, error = ?
         WHERE status = 'running' AND {key_col} IN ({placeholders})
        """,
        [now, STALE_ERROR, *keys],
    )
    return int(cur.rowcount)


def sweep_stale_runs(minutes: int = 30, *, db_path: Path | None = None) -> dict[str, int]:
    """Close `running` rows left behind by a process that died: graph_runs,
    node_runs, cio_runs, spans and llm_calls become `failed` with
    error=STALE_ERROR. A node row is stale when its pid is gone or it is older
    than `minutes`; a graph run when it is older than the window or every pid
    its nodes ran under is dead; a CIO cycle when it is older than the window
    and opened no span inside it (a live cycle keeps opening spans); spans
    and LLM calls follow the window or a parent swept in the same pass.
    Cheap: five small scans. Called by the Live tab on each poll and by the
    CIO dispatcher at cycle start. Returns rows closed per table."""
    init_db(db_path)
    cutoff = (datetime.now(UTC) - timedelta(minutes=int(minutes))).isoformat()
    now = _now_iso()
    alive: dict[int, bool] = {}

    def _alive(pid: Any) -> bool:
        key = int(pid)
        if key not in alive:
            alive[key] = _pid_alive(key)
        return alive[key]

    with _connect(db_path) as conn:
        stale_nodes = [
            r["id"]
            for r in conn.execute(
                "SELECT id, pid, started_at FROM node_runs WHERE status = 'running'"
            )
            if r["started_at"] < cutoff or (r["pid"] is not None and not _alive(r["pid"]))
        ]
        stale_runs: list[str] = []
        for r in conn.execute("SELECT run_id, started_at FROM graph_runs WHERE status = 'running'"):
            if r["started_at"] < cutoff:
                stale_runs.append(r["run_id"])
                continue
            pids = [
                p["pid"]
                for p in conn.execute(
                    "SELECT DISTINCT pid FROM node_runs WHERE run_id = ? AND pid IS NOT NULL",
                    (r["run_id"],),
                )
            ]
            if pids and not any(_alive(p) for p in pids):
                stale_runs.append(r["run_id"])
        stale_cio = [
            r["run_id"]
            for r in conn.execute(
                """
                SELECT c.run_id FROM cio_runs c
                 WHERE c.status = 'running' AND c.started_at < ?
                   AND NOT EXISTS (
                       SELECT 1 FROM spans s
                        WHERE s.parent_kind = 'cio_run' AND s.parent_id = c.run_id
                          AND s.started_at >= ?)
                """,
                (cutoff, cutoff),
            )
        ]
        swept_parents = set(stale_runs) | set(stale_cio)
        stale_spans = [
            r["id"]
            for r in conn.execute(
                "SELECT id, parent_id, started_at FROM spans WHERE status = 'running'"
            )
            if r["started_at"] < cutoff or r["parent_id"] in swept_parents
        ]
        stale_calls = [
            r["id"]
            for r in conn.execute("SELECT id, run_id, ts FROM llm_calls WHERE status = 'running'")
            if r["ts"] < cutoff or (r["run_id"] is not None and r["run_id"] in swept_parents)
        ]
        counts = {
            "node_runs": _mark_stale(conn, "node_runs", "id", stale_nodes, now),
            "graph_runs": _mark_stale(conn, "graph_runs", "run_id", stale_runs, now),
            "cio_runs": _mark_stale(conn, "cio_runs", "run_id", stale_cio, now),
            "spans": _mark_stale(conn, "spans", "id", stale_spans, now),
            "llm_calls": _mark_stale(conn, "llm_calls", "id", stale_calls, now),
        }
    return counts


def node_medians(limit: int = 60, *, db_path: Path | None = None) -> dict[str, float]:
    """Median duration in seconds per node, plus `run` for the whole graph,
    over the last `limit` completed graph runs — the live view's ETA baseline.
    Empty when nothing has completed yet."""
    if not Path(db_path or DB_PATH).exists():
        return {}
    with _connect(db_path) as conn:
        runs = conn.execute(
            """
            SELECT run_id, duration_s FROM graph_runs
             WHERE status = 'completed' ORDER BY started_at DESC LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        if not runs:
            return {}
        ids = [r["run_id"] for r in runs]
        placeholders = ",".join("?" for _ in ids)
        nodes = conn.execute(
            f"""
            SELECT node, duration_s FROM node_runs
             WHERE run_id IN ({placeholders}) AND status = 'completed'
               AND duration_s IS NOT NULL
            """,
            ids,
        ).fetchall()
    by_node: dict[str, list[float]] = {}
    for r in nodes:
        by_node.setdefault(r["node"], []).append(float(r["duration_s"]))
    out = {node: round(statistics.median(v), 1) for node, v in sorted(by_node.items())}
    run_durations = [float(r["duration_s"]) for r in runs if r["duration_s"] is not None]
    if run_durations:
        out["run"] = round(statistics.median(run_durations), 1)
    return out


# --- Schema v9: the snapshot (one shape for live and replay) -----------------


def next_heartbeat(now: datetime | None = None) -> str:
    """ISO local time of the next scheduled CIO heartbeat (HEARTBEAT_HOURS_LOCAL)."""
    now = now or datetime.now().astimezone()
    for day_offset in (0, 1):
        day = now + timedelta(days=day_offset)
        for hour in sorted(HEARTBEAT_HOURS_LOCAL):
            candidate = day.replace(hour=hour, minute=0, second=0, microsecond=0)
            if candidate > now:
                return candidate.isoformat(timespec="minutes")
    return now.isoformat(timespec="minutes")


def _empty_vitals() -> dict:
    return {
        "runs_today": 0,
        "cost_today_usd": 0.0,
        "failed_24h": 0,
        "degraded_24h": 0,
        "active": 0,
        "next_heartbeat": next_heartbeat(),
    }


def _vitals(conn: sqlite3.Connection) -> dict:
    today = datetime.now(UTC).date().isoformat()
    day_ago = (datetime.now(UTC) - timedelta(hours=24)).isoformat()
    runs_today = conn.execute(
        "SELECT COUNT(*) AS n FROM graph_runs WHERE substr(started_at, 1, 10) = ?", (today,)
    ).fetchone()["n"]
    cost = conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0.0) AS c FROM node_runs WHERE substr(started_at, 1, 10) = ?",
        (today,),
    ).fetchone()["c"]
    failed = conn.execute(
        "SELECT COUNT(*) AS n FROM graph_runs WHERE status = 'failed' AND started_at >= ?",
        (day_ago,),
    ).fetchone()["n"]
    # Degraded = finished, but an agent failed or logged a run-scoped soft
    # error (same derivation as Mission Control's runs table).
    degraded = conn.execute(
        """
        SELECT COUNT(*) AS n FROM graph_runs g
         WHERE g.status = 'completed' AND g.started_at >= ?
           AND (EXISTS (SELECT 1 FROM node_runs n WHERE n.run_id = g.run_id AND n.status = 'failed')
                OR EXISTS (SELECT 1 FROM errors e WHERE e.run_id = g.run_id))
        """,
        (day_ago,),
    ).fetchone()["n"]
    active = conn.execute(
        "SELECT COUNT(*) AS n FROM graph_runs WHERE status = 'running'"
    ).fetchone()["n"]
    return {
        "runs_today": int(runs_today),
        "cost_today_usd": round(float(cost or 0.0), 4),
        "failed_24h": int(failed),
        "degraded_24h": int(degraded),
        "active": int(active),
        "next_heartbeat": next_heartbeat(),
    }


def _run_detail(conn: sqlite3.Connection, run: dict, *, include_messages: bool) -> dict:
    """One graph run with its node rows, each node's LLM calls and — when
    `include_messages` — every call's full request messages and response."""
    run_id = run["run_id"]
    nodes = [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM node_runs WHERE run_id = ? ORDER BY started_at, id", (run_id,)
        )
    ]
    calls = [
        dict(r)
        for r in conn.execute("SELECT * FROM llm_calls WHERE run_id = ? ORDER BY id", (run_id,))
    ]
    messages_by_call: dict[int, list[dict]] = {}
    if include_messages and calls:
        placeholders = ",".join("?" for _ in calls)
        for r in conn.execute(
            f"""
            SELECT m.call_id, m.role, b.content
              FROM llm_messages m JOIN prompt_blobs b ON b.sha256 = m.sha256
             WHERE m.call_id IN ({placeholders})
             ORDER BY m.call_id, m.seq
            """,
            [c["id"] for c in calls],
        ):
            messages_by_call.setdefault(r["call_id"], []).append(
                {"role": r["role"], "text": r["content"]}
            )
    calls_by_node: dict[str, list[dict]] = {}
    for c in calls:
        entry = {
            "id": c["id"],
            "model": c["model"],
            "status": c.get("status") or "completed",
            "attempt": c.get("attempt") or 1,
            "started_at": c["ts"],
            "ended_at": c.get("ended_at"),
            "latency_s": c["latency_s"],
            "tokens_in": c["tokens_in"],
            "tokens_out": c["tokens_out"],
            "cost_usd": c["cost_usd"],
            "error": c.get("error"),
            "prompt_excerpt": c["prompt_excerpt"],
            "response_excerpt": c["response_excerpt"],
        }
        if include_messages:
            msgs = messages_by_call.get(c["id"], [])
            entry["messages"] = [m for m in msgs if m["role"] != "assistant"]
            replies = [m["text"] for m in msgs if m["role"] == "assistant"]
            entry["response"] = replies[-1] if replies else None
        calls_by_node.setdefault(c["node"], []).append(entry)
    node_entries = [
        {
            "id": n["id"],
            "node": n["node"],
            "status": n["status"],
            "started_at": n["started_at"],
            "ended_at": n["ended_at"],
            "duration_s": n["duration_s"],
            "error": n["error"],
            "pid": n.get("pid"),
            "tokens_in": n["tokens_in"],
            "tokens_out": n["tokens_out"],
            "cost_usd": n["cost_usd"],
            "n_calls": n["n_calls"],
            "calls": calls_by_node.get(n["node"], []),
        }
        for n in nodes
    ]
    errors = [
        dict(r)
        for r in conn.execute("SELECT * FROM errors WHERE run_id = ? ORDER BY ts", (run_id,))
    ]
    return {
        "run_id": run_id,
        "ticker": run.get("ticker"),
        "thesis": run.get("thesis"),
        "source": run.get("source") or "",
        "status": run.get("status"),
        "started_at": run.get("started_at"),
        "ended_at": run.get("ended_at"),
        "duration_s": run.get("duration_s"),
        "confidence": run.get("confidence"),
        "error": run.get("error"),
        "nodes": node_entries,
        "errors": errors,
    }


def _cio_detail(conn: sqlite3.Connection, cycle: dict) -> dict:
    run_id = cycle["run_id"]
    steps = [
        _span_dict(r)
        for r in conn.execute(
            "SELECT * FROM spans WHERE parent_kind = 'cio_run' AND parent_id = ? ORDER BY id",
            (run_id,),
        )
    ]
    actions = [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM cio_actions WHERE cio_run_id = ? ORDER BY id", (run_id,)
        )
    ]
    return {**cycle, "steps": steps, "actions": actions}


def _empty_snapshot() -> dict:
    return {
        "now": _now_iso(),
        "poll_s": LIVE_POLL_S,
        "vitals": _empty_vitals(),
        "medians": {},
        "cio": [],
        "runs": [],
        "recent": [],
    }


def live_snapshot(
    *,
    recent_limit: int = 20,
    cio_limit: int = 5,
    sweep: bool = True,
    db_path: Path | None = None,
) -> dict:
    """The single state document the Live tab renders (plan §3.3): server
    time, vitals, node medians, the running (then most recent) CIO cycles with
    their steps and actions, every ACTIVE graph run with nodes, calls and full
    messages, and the last `recent_limit` runs with per-node status. Replay
    feeds the same renderer from `run_snapshot`. Runs `sweep_stale_runs` first
    unless `sweep=False`. Never creates the DB file."""
    if not Path(db_path or DB_PATH).exists():
        return _empty_snapshot()
    if sweep:
        try:
            sweep_stale_runs(db_path=db_path)
        except Exception as exc:
            logger.warning(f"[telemetry] stale sweep failed: {exc}")
    with _connect(db_path) as conn:
        try:
            vitals = _vitals(conn)
            active = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM graph_runs WHERE status = 'running' ORDER BY started_at"
                )
            ]
            runs = [_run_detail(conn, r, include_messages=True) for r in active]
            cycles = [
                dict(r)
                for r in conn.execute(
                    """
                    SELECT * FROM cio_runs
                     ORDER BY (status = 'running') DESC, started_at DESC LIMIT ?
                    """,
                    (int(cio_limit),),
                )
            ]
            cio = [_cio_detail(conn, c) for c in cycles]
        except sqlite3.OperationalError as exc:  # pre-v9 file nothing has written to yet
            logger.warning(f"[telemetry] live_snapshot on a pre-v9 DB: {exc}")
            return _empty_snapshot()
    recent = recent_runs(limit=recent_limit, db_path=db_path)
    statuses = node_status_by_run([r["run_id"] for r in recent], db_path=db_path)
    for r in recent:
        r["nodes"] = statuses.get(r["run_id"], {})
    return {
        "now": _now_iso(),
        "poll_s": LIVE_POLL_S,
        "vitals": vitals,
        "medians": node_medians(db_path=db_path),
        "cio": cio,
        "runs": runs,
        "recent": recent,
    }


def run_snapshot(run_id: str, *, db_path: Path | None = None) -> dict | None:
    """One graph run in the snapshot's `runs[]` shape with full messages, plus
    `cio` — the CIO decision that triggered or reused it, if any. The Runs
    tab's inspector drawer and replay input. None when the run is unknown."""
    if not run_id or not Path(db_path or DB_PATH).exists():
        return None
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM graph_runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        try:
            detail = _run_detail(conn, dict(row), include_messages=True)
        except sqlite3.OperationalError:
            return None
    detail["cio"] = cio_action_for_run(run_id, db_path=db_path)
    return detail
