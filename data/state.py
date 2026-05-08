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

What we DO NOT record:
  - Per-LLM-call tokens / USD cost / full prompt / response. **LangSmith**
    auto-instruments these when LANGSMITH_TRACING=true; duplicating that
    work in our SQLite would just diverge. The dashboard's Mission Control
    panel will deep-link to LangSmith for per-call detail; state.db is for
    aggregate-and-historical queries.

Run-ID propagation via `contextvars` — the graph wrapper sets
`current_run_id` once per ainvoke, and `_safe_node` reads it without
touching the FinaqState shape (keeps the state clean).
"""

from __future__ import annotations

import contextvars
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DB_PATH = Path("data_cache/state.db")
SCHEMA_VERSION = 4  # Step 11.19 — per-CIO-call telemetry: model_used, tokens, cost_usd, latency_s

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


def new_node_telemetry() -> dict:
    """Return a fresh accumulator dict for a single node invocation.
    Caller binds it to `node_telemetry_var` for the node's lifetime."""
    return {"tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "n_calls": 0}


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
    confidence  TEXT                            -- low | medium | high (from synthesis)
);

CREATE TABLE IF NOT EXISTS node_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT,
    node        TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    duration_s  REAL,
    status      TEXT NOT NULL,                  -- completed | failed
    error       TEXT,
    -- Step 10c.8 (schema v2) — per-node LLM telemetry. Populated by the
    -- ContextVar accumulator in `utils.openrouter`. Nodes that don't make
    -- LLM calls (load_thesis, monte_carlo) leave these at 0.
    tokens_in   INTEGER NOT NULL DEFAULT 0,
    tokens_out  INTEGER NOT NULL DEFAULT 0,
    cost_usd    REAL    NOT NULL DEFAULT 0.0,
    n_calls     INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (run_id) REFERENCES graph_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_node_runs_run_id ON node_runs(run_id);
CREATE INDEX IF NOT EXISTS idx_graph_runs_started_at ON graph_runs(started_at);

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
    FOREIGN KEY (cio_run_id) REFERENCES cio_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_cio_actions_ts ON cio_actions(ts);
CREATE INDEX IF NOT EXISTS idx_cio_actions_cio_run_id ON cio_actions(cio_run_id);
CREATE INDEX IF NOT EXISTS idx_cio_actions_ticker_thesis ON cio_actions(ticker, thesis, ts);
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

        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
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
    db_path: Path | None = None,
) -> str:
    """Insert a graph_runs row in the `running` state and return the run_id.
    Caller must call `finish_graph_run` after the run completes."""
    init_db(db_path)
    run_id = _new_run_id()
    started_at = _now_iso()
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO graph_runs (run_id, ticker, thesis, started_at, status)
            VALUES (?, ?, ?, ?, 'running')
            """,
            (run_id, ticker, thesis, started_at),
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


# --- Queries ----------------------------------------------------------------


def recent_runs(limit: int = 20, *, db_path: Path | None = None) -> list[dict]:
    """Most-recent-first list of graph runs, with completed-node count."""
    if not Path(db_path or DB_PATH).exists():
        return []
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT g.*, COUNT(n.id) AS node_runs_count,
                   SUM(CASE WHEN n.status = 'failed' THEN 1 ELSE 0 END) AS failed_nodes
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
    db_path: Path | None = None,
) -> int:
    """Append a cio_actions row. Returns the row id.

    `action` ∈ {drill, reuse, dismiss}. `confidence` ∈ {low, medium, high}.
    `cio_run_id` is the parent cycle id from `start_cio_run` (None ok for
    standalone unit tests).

    Step 11.19 — `model_used`, tokens, cost_usd, and latency_s capture the
    LLM call that produced this decision. All zero/None for gate-shortcut
    decisions (yo-yo guard) since no LLM call fired.
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
                model_used, tokens_in, tokens_out, cost_usd, latency_s
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cio_run_id, _now_iso(), trigger, ticker, thesis, action, rationale,
                drill_run_id, reuse_run_id, confidence, decision_json,
                model_used,
                int(tokens_in or 0), int(tokens_out or 0),
                float(cost_usd or 0.0), float(latency_s or 0.0),
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
    db_path: Path | None = None,
) -> list[dict]:
    """Most-recent-first list of CIO decisions. Optionally scope to a
    specific (ticker, thesis) pair — used by the planner's cooldown gate."""
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
    if conds:
        sql += " WHERE " + " AND ".join(conds)
    sql += " ORDER BY ts DESC LIMIT ?"
    args.append(limit)
    with _connect(db_path) as conn:
        rows = conn.execute(sql, args).fetchall()
    return [dict(r) for r in rows]


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
