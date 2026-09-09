"""Persistent halo-graph store (Phase 2 Discovery; ARCHITECTURE §13.3).

The nodes + edges of a discovered thesis live in two `state.db` tables
(`graph_nodes`, `graph_edges`). Their DDL + migration live in `data.state`
(schema v8); this module is the domain API over them: save a discovered
graph, read it back, and traverse neighbours.

Chosen over Pinecone / Notion / Postgres-on-Supabase in §13.3: a halo graph is
structured, related, versioned, traversed data of a few hundred nodes for one
user — SQLite is ample, transactional, and adds no dependency. Traversal uses
SQLite recursive CTEs, so there is no graph library to pull in.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from data import state
from utils import logger
from utils.schemas import Evidence, GraphEdge, GraphNode

# Traversal is capped so a pathological/among-cycles graph can't spin. A halo
# graph is shallow by construction; nobody needs 5-hop reach.
MAX_HOPS = 4


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    """Open a connection to the (migrated) state.db.

    Mirrors `data.state._connect` but resolves `state.DB_PATH` at call time so
    the conftest monkeypatch to a tmp DB is respected. Calls `state.init_db`
    first so the graph tables exist even on a fresh database.
    """
    state.init_db(db_path)
    target = Path(db_path) if db_path is not None else state.DB_PATH
    conn = sqlite3.connect(target, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _db_exists(db_path: Path | None) -> bool:
    return Path(db_path if db_path is not None else state.DB_PATH).exists()


def save_graph(
    thesis_slug: str,
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    *,
    db_path: Path | None = None,
) -> None:
    """Replace the stored graph for `thesis_slug` with these nodes + edges.

    Delete-then-insert (in one transaction) so a re-run of Discovery for the
    same slug is idempotent — a previously larger universe leaves no orphan
    rows. Stores every edge, grounded or not, for audit; callers filter on
    `grounded` when they only want the thesis-worthy ones.
    """
    now = _now_iso()
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM graph_edges WHERE thesis_slug = ?", (thesis_slug,))
        conn.execute("DELETE FROM graph_nodes WHERE thesis_slug = ?", (thesis_slug,))
        conn.executemany(
            """INSERT INTO graph_nodes (thesis_slug, ticker, name, first_seen, last_seen)
               VALUES (?, ?, ?, ?, ?)""",
            [
                (thesis_slug, n.ticker, n.name, n.first_seen or now, n.last_seen or now)
                for n in nodes
            ],
        )
        conn.executemany(
            """INSERT INTO graph_edges (thesis_slug, from_ticker, to_ticker, type, note,
                   confidence, grounded, evidence_json, as_of)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    thesis_slug,
                    e.from_,
                    e.to,
                    e.type,
                    e.note,
                    float(e.confidence),
                    1 if e.grounded else 0,
                    json.dumps([ev.model_dump() for ev in e.evidence]),
                    e.as_of or now,
                )
                for e in edges
            ],
        )
    logger.info(f"[graph] saved {len(nodes)} nodes + {len(edges)} edges for {thesis_slug!r}")


def _row_to_edge(row: sqlite3.Row) -> GraphEdge:
    try:
        raw_ev = json.loads(row["evidence_json"] or "[]")
    except (json.JSONDecodeError, TypeError):
        raw_ev = []
    evidence = []
    for ev in raw_ev:
        try:
            evidence.append(Evidence.model_validate(ev))
        except Exception:  # noqa: BLE001 — tolerate a legacy/malformed evidence blob
            continue
    return GraphEdge(
        **{"from": row["from_ticker"]},
        to=row["to_ticker"],
        type=row["type"],
        note=row["note"] or "",
        confidence=row["confidence"] or 0.0,
        grounded=bool(row["grounded"]),
        evidence=evidence,
        thesis_slug=row["thesis_slug"],
        as_of=row["as_of"] or "",
    )


def get_edges(
    thesis_slug: str,
    *,
    grounded_only: bool = False,
    db_path: Path | None = None,
) -> list[GraphEdge]:
    """All edges for a thesis, highest-confidence first. `grounded_only`
    restricts to edges that cleared the grounding threshold."""
    if not _db_exists(db_path):
        return []
    sql = "SELECT * FROM graph_edges WHERE thesis_slug = ?"
    if grounded_only:
        sql += " AND grounded = 1"
    sql += " ORDER BY confidence DESC, from_ticker, to_ticker"
    try:
        with _connect(db_path) as conn:
            return [_row_to_edge(r) for r in conn.execute(sql, (thesis_slug,))]
    except sqlite3.OperationalError:  # pre-migration DB — no graph tables yet
        return []


def get_nodes(thesis_slug: str, *, db_path: Path | None = None) -> list[GraphNode]:
    """All nodes for a thesis, ticker-sorted."""
    if not _db_exists(db_path):
        return []
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM graph_nodes WHERE thesis_slug = ? ORDER BY ticker",
                (thesis_slug,),
            )
            return [
                GraphNode(
                    ticker=r["ticker"],
                    name=r["name"] or "",
                    thesis_slug=r["thesis_slug"],
                    first_seen=r["first_seen"],
                    last_seen=r["last_seen"],
                )
                for r in rows
            ]
    except sqlite3.OperationalError:
        return []


def neighbors(
    ticker: str,
    *,
    hops: int = 1,
    thesis_slug: str | None = None,
    grounded_only: bool = True,
    db_path: Path | None = None,
) -> list[str]:
    """Tickers within `hops` of `ticker`, following edges in either direction.

    Uses a recursive CTE; `UNION` (not `UNION ALL`) plus the depth guard makes
    it terminate on cycles. Excludes the seed itself. Scoped to one thesis when
    `thesis_slug` is given; follows only grounded edges by default. `hops` is
    clamped to [1, MAX_HOPS].
    """
    if not _db_exists(db_path):
        return []
    hops = max(1, min(int(hops), MAX_HOPS))
    seed = ticker.upper()
    grounded_clause = "AND e.grounded = 1" if grounded_only else ""
    slug_clause = "AND e.thesis_slug = :slug" if thesis_slug else ""
    sql = f"""
        WITH RECURSIVE reach(node, depth) AS (
            SELECT :seed, 0
            UNION
            SELECT CASE WHEN e.from_ticker = r.node THEN e.to_ticker
                        ELSE e.from_ticker END,
                   r.depth + 1
            FROM reach r
            JOIN graph_edges e
              ON (e.from_ticker = r.node OR e.to_ticker = r.node)
              {grounded_clause} {slug_clause}
            WHERE r.depth < :hops
        )
        SELECT DISTINCT node FROM reach WHERE node != :seed ORDER BY node
    """
    params: dict[str, object] = {"seed": seed, "hops": hops}
    if thesis_slug:
        params["slug"] = thesis_slug
    try:
        with _connect(db_path) as conn:
            return [r["node"] for r in conn.execute(sql, params)]
    except sqlite3.OperationalError:
        return []
