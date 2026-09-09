"""Unit tests for the persistent halo-graph store (`data/graph.py`).

All tests pass an explicit `db_path` under `tmp_path` so they exercise a fresh
SQLite file and never touch the real `data_cache/state.db`.
"""

from __future__ import annotations

from data import graph
from utils.schemas import Evidence, GraphEdge, GraphNode

_TS = "2026-09-09T00:00:00+00:00"


def _node(ticker: str, slug: str = "adhoc_t", name: str = "") -> GraphNode:
    return GraphNode(
        ticker=ticker,
        name=name or ticker,
        thesis_slug=slug,
        first_seen=_TS,
        last_seen=_TS,
    )


def _edge(
    frm: str,
    to: str,
    slug: str = "adhoc_t",
    rel_type: str = "supplier",
    conf: float = 0.5,
    grounded: bool = True,
    evidence: list[Evidence] | None = None,
) -> GraphEdge:
    return GraphEdge(
        **{"from": frm},
        to=to,
        type=rel_type,  # type: ignore[arg-type]
        note=f"{frm}->{to}",
        confidence=conf,
        grounded=grounded,
        evidence=evidence or [],
        thesis_slug=slug,
        as_of=_TS,
    )


def test_save_and_get_roundtrip(tmp_path):
    db = tmp_path / "g.db"
    ev = [Evidence(source="edgar", accession="acc1", excerpt="A buys from B")]
    graph.save_graph(
        "adhoc_t", [_node("A"), _node("B")], [_edge("A", "B", conf=0.8, evidence=ev)], db_path=db
    )

    edges = graph.get_edges("adhoc_t", db_path=db)
    assert len(edges) == 1
    assert edges[0].from_ == "A" and edges[0].to == "B"
    assert edges[0].confidence == 0.8 and edges[0].grounded is True
    assert edges[0].evidence[0].accession == "acc1"

    nodes = graph.get_nodes("adhoc_t", db_path=db)
    assert [n.ticker for n in nodes] == ["A", "B"]


def test_grounded_only_and_confidence_ordering(tmp_path):
    db = tmp_path / "g.db"
    edges = [
        _edge("A", "B", conf=0.3, grounded=True),
        _edge("A", "C", conf=0.9, grounded=True),
        _edge("A", "D", conf=0.0, grounded=False),
    ]
    graph.save_graph("adhoc_t", [_node(x) for x in "ABCD"], edges, db_path=db)

    assert [e.to for e in graph.get_edges("adhoc_t", db_path=db)] == ["C", "B", "D"]
    assert [e.to for e in graph.get_edges("adhoc_t", grounded_only=True, db_path=db)] == ["C", "B"]


def test_save_replaces_prior_graph(tmp_path):
    db = tmp_path / "g.db"
    graph.save_graph("adhoc_t", [_node("A"), _node("B")], [_edge("A", "B")], db_path=db)
    # Re-run with a smaller universe: no orphan edges/nodes left behind.
    graph.save_graph("adhoc_t", [_node("A")], [], db_path=db)
    assert graph.get_edges("adhoc_t", db_path=db) == []
    assert [n.ticker for n in graph.get_nodes("adhoc_t", db_path=db)] == ["A"]


def test_neighbors_hops_and_grounding(tmp_path):
    db = tmp_path / "g.db"
    # A—B—C grounded; C—D behind an ungrounded edge.
    edges = [
        _edge("A", "B"),
        _edge("B", "C"),
        _edge("C", "D", conf=0.0, grounded=False),
    ]
    graph.save_graph("adhoc_t", [_node(x) for x in "ABCD"], edges, db_path=db)

    assert graph.neighbors("A", hops=1, db_path=db) == ["B"]
    assert graph.neighbors("A", hops=2, db_path=db) == ["B", "C"]
    # D is unreachable following grounded edges only, even at higher hops.
    assert graph.neighbors("A", hops=3, db_path=db) == ["B", "C"]
    # Following all edges, D is reachable at hop 3.
    assert graph.neighbors("A", hops=3, grounded_only=False, db_path=db) == ["B", "C", "D"]


def test_neighbors_scoped_to_thesis(tmp_path):
    db = tmp_path / "g.db"
    graph.save_graph(
        "adhoc_x",
        [_node("A", "adhoc_x"), _node("B", "adhoc_x")],
        [_edge("A", "B", "adhoc_x")],
        db_path=db,
    )
    graph.save_graph(
        "adhoc_y",
        [_node("A", "adhoc_y"), _node("Z", "adhoc_y")],
        [_edge("A", "Z", "adhoc_y")],
        db_path=db,
    )
    assert graph.neighbors("A", thesis_slug="adhoc_x", db_path=db) == ["B"]
    assert graph.neighbors("A", thesis_slug="adhoc_y", db_path=db) == ["Z"]
    # Unscoped, the seed reaches both theses' neighbours.
    assert graph.neighbors("A", db_path=db) == ["B", "Z"]


def test_neighbors_clamps_and_excludes_seed(tmp_path):
    db = tmp_path / "g.db"
    graph.save_graph("adhoc_t", [_node("A"), _node("B")], [_edge("A", "B")], db_path=db)
    # hops far above MAX_HOPS is clamped, not rejected; seed never in output.
    out = graph.neighbors("A", hops=99, db_path=db)
    assert "A" not in out and out == ["B"]


def test_missing_db_returns_empty(tmp_path):
    db = tmp_path / "does_not_exist.db"
    assert graph.get_edges("adhoc_t", db_path=db) == []
    assert graph.get_nodes("adhoc_t", db_path=db) == []
    assert graph.neighbors("A", db_path=db) == []


def test_save_tolerates_duplicate_pair(tmp_path):
    db = tmp_path / "g.db"
    # Two edges for the same (slug, from, to) — the second is ignored, not an
    # IntegrityError that would roll back the whole graph (nodes included).
    edges = [_edge("A", "B", rel_type="supplier"), _edge("A", "B", rel_type="customer")]
    graph.save_graph("adhoc_t", [_node("A"), _node("B")], edges, db_path=db)
    assert len(graph.get_edges("adhoc_t", db_path=db)) == 1
    assert [n.ticker for n in graph.get_nodes("adhoc_t", db_path=db)] == ["A", "B"]
