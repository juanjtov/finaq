"""Unit + render tests for the Halo graph page (`ui/pages/halo_graph.py`).

Same split as the other page tests (see test_dashboard_smoke.py):
  - Pure view-model / mermaid helpers tested directly — fast, deterministic.
  - `AppTest` render smoke tests for the fallback (thesis-file) path and the
    rich (graph-store) path, asserting no exception + the right source label.

The autouse `_isolated_state_db` fixture (conftest.py) already points
`data.state.DB_PATH` at a fresh tmp file, so writing a graph with
`graph.save_graph(...)` lands in the same DB the page reads back in-process.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from streamlit.testing.v1 import AppTest

from data import graph
from utils.schemas import Evidence, GraphEdge, GraphNode, Thesis

hg = importlib.import_module("ui.pages.halo_graph")

PAGES_DIR = Path(__file__).parents[1] / "ui" / "pages"
DASHBOARD_TIMEOUT_S = 30


# --- _safe_id / _safe_label -------------------------------------------------


def test_safe_id_sanitizes_dotted_and_dashed_tickers():
    assert hg._safe_id("BRK.B") == "BRK_B"
    assert hg._safe_id("RDS-A") == "RDS_A"
    assert hg._safe_id("NVDA") == "NVDA"
    assert hg._safe_id("") == "_"


def test_safe_label_escapes_html_and_ellipsises():
    # HTML-significant chars are escaped (not dropped) so they render literally
    # inside the quoted, htmlLabels mermaid node label — including `&`, which the
    # old strip-only version passed through raw (an invalid entity like `&P`).
    out = hg._safe_label('AT&T "Inc" <x>')
    assert "&amp;" in out  # `&` preserved via escaping, not silently kept raw
    assert "&quot;" in out and "&lt;" in out and "&gt;" in out
    assert '"' not in out and "<" not in out and ">" not in out
    ellipsised = hg._safe_label("A" * 40, limit=10)
    assert len(ellipsised) == 10 and ellipsised.endswith("…")


# --- _mermaid ---------------------------------------------------------------


def _sample():
    nodes = [
        hg.VizNode("NVDA", "NVIDIA Corp", True),
        hg.VizNode("VRT", "Vertiv", False),
        hg.VizNode("AMD", "Advanced Micro", False),
    ]
    edges = [
        hg.VizEdge("NVDA", "VRT", "supplier", "cooling", 0.8, True, []),
        hg.VizEdge("NVDA", "AMD", "competitor", "gpu rivalry", 0.1, False, []),
    ]
    return nodes, edges


def test_mermaid_marks_anchor_and_members():
    nodes, edges = _sample()
    code = hg._mermaid(nodes, edges, show_ungrounded=False)
    assert "classDef anchor" in code and "classDef member" in code
    assert "class NVDA anchor;" in code
    # Members are grouped into one class statement (order preserved).
    assert "class VRT,AMD member;" in code


def test_mermaid_hides_ungrounded_by_default_and_shows_when_toggled():
    nodes, edges = _sample()
    hidden = hg._mermaid(nodes, edges, show_ungrounded=False)
    # Grounded edge: solid arrow, type + confidence in the label.
    assert 'NVDA -->|"supplier · 0.80"| VRT' in hidden
    assert "-.->" not in hidden  # no dashed (ungrounded) edge drawn
    assert "competitor" not in hidden  # the pruned edge is absent entirely

    shown = hg._mermaid(nodes, edges, show_ungrounded=True)
    assert 'NVDA -.->|"competitor · 0.10"| AMD' in shown
    # One linkStyle per drawn edge, indexed from 0 in declaration order.
    assert "linkStyle 0 " in shown and "linkStyle 1 " in shown


def test_mermaid_omits_confidence_when_unknown():
    # confidence None (thesis-file edge) → label is the bare type, no separator.
    nodes = [hg.VizNode("AAA", "Alpha", True), hg.VizNode("BBB", "Beta", False)]
    edges = [hg.VizEdge("AAA", "BBB", "peer", "", None, True, [])]
    code = hg._mermaid(nodes, edges, show_ungrounded=False)
    assert '|"peer"|' in code
    assert " · " not in code


def test_mermaid_disambiguates_colliding_safe_ids():
    # BRK.B and BRK-B both sanitise to BRK_B — two DISTINCT companies must not
    # collapse into one mermaid node. The second gets a numeric suffix, and the
    # edge connects the two distinct ids (not a self-loop).
    nodes = [
        hg.VizNode("BRK.B", "Berkshire B", True),
        hg.VizNode("BRK-B", "Berkshire dash", False),
    ]
    edges = [hg.VizEdge("BRK.B", "BRK-B", "peer", "", 0.5, True, [])]
    code = hg._mermaid(nodes, edges, show_ungrounded=False)
    assert 'BRK_B["BRK.B<br/>Berkshire B"]' in code
    assert 'BRK_B_2["BRK-B<br/>Berkshire dash"]' in code
    assert 'BRK_B -->|"peer · 0.50"| BRK_B_2' in code


# --- _build_model -----------------------------------------------------------


def test_build_model_prefers_graph_store(monkeypatch, tmp_path):
    """When a graph is persisted for the slug, the page reads confidence,
    grounded flags, evidence, AND the pruned edges from the store."""
    from data import state as state_db

    monkeypatch.setattr(state_db, "DB_PATH", tmp_path / "g.db")
    slug = "adhoc_test_topic"
    now = "2026-09-09T00:00:00+00:00"
    nodes = [
        GraphNode(ticker=t, name=n, thesis_slug=slug, first_seen=now, last_seen=now)
        for t, n in (("NVDA", "NVIDIA"), ("VRT", "Vertiv"), ("AMD", "AMD"))
    ]
    edges = [
        GraphEdge(
            **{"from": "NVDA"},
            to="VRT",
            type="supplier",
            note="cooling",
            confidence=0.8,
            grounded=True,
            evidence=[Evidence(source="edgar", accession="0001", excerpt="…")],
            thesis_slug=slug,
            as_of=now,
        ),
        GraphEdge(
            **{"from": "NVDA"},
            to="AMD",
            type="competitor",
            note="",
            confidence=0.1,
            grounded=False,
            evidence=[],
            thesis_slug=slug,
            as_of=now,
        ),
    ]
    graph.save_graph(slug, nodes, edges)  # uses the monkeypatched DB_PATH

    thesis = Thesis(
        name="Test",
        summary="s",
        anchor_tickers=["NVDA"],
        universe=["NVDA", "VRT", "AMD"],
        relationships=[{"from": "NVDA", "to": "VRT", "type": "supplier"}],
    )
    vnodes, vedges, from_store = hg._build_model(slug, thesis)

    assert from_store is True
    assert len(vedges) == 2  # pruned edge kept for audit
    grounded = [e for e in vedges if e.grounded]
    assert len(grounded) == 1
    assert grounded[0].confidence == 0.8
    assert grounded[0].evidence and grounded[0].evidence[0]["source"] == "edgar"
    assert any(n.is_anchor and n.ticker == "NVDA" for n in vnodes)
    # Company names come from the store, not the bare ticker.
    assert next(n for n in vnodes if n.ticker == "VRT").name == "Vertiv"


def test_build_model_falls_back_to_thesis_file(monkeypatch, tmp_path):
    """No graph persisted → build from the thesis file. Those relationships
    are the grounded subset by definition; confidence is unknown."""
    from data import state as state_db

    monkeypatch.setattr(state_db, "DB_PATH", tmp_path / "empty.db")
    thesis = Thesis(
        name="Curated",
        summary="s",
        anchor_tickers=["AAA"],
        universe=["AAA", "BBB"],
        relationships=[{"from": "AAA", "to": "BBB", "type": "peer"}],
    )
    vnodes, vedges, from_store = hg._build_model("curated_slug", thesis)

    assert from_store is False
    assert len(vnodes) == 2 and len(vedges) == 1
    assert vedges[0].grounded is True
    assert vedges[0].confidence is None


def test_build_model_falls_back_when_store_has_nodes_but_no_edges(monkeypatch, tmp_path):
    """A discovery run can persist a universe but ground zero edges. The store
    then has nodes but no edges — the page must fall back to the thesis file's
    real relationships rather than render an empty graph."""
    from data import state as state_db

    monkeypatch.setattr(state_db, "DB_PATH", tmp_path / "nodesonly.db")
    slug = "curated_with_nodes"
    now = "2026-09-09T00:00:00+00:00"
    graph.save_graph(
        slug,
        [
            GraphNode(ticker=t, name=f"{t} Inc", thesis_slug=slug, first_seen=now, last_seen=now)
            for t in ("AAA", "BBB")
        ],
        [],  # universe persisted, but nothing grounded
    )
    thesis = Thesis(
        name="Curated",
        summary="s",
        anchor_tickers=["AAA"],
        universe=["AAA", "BBB"],
        relationships=[{"from": "AAA", "to": "BBB", "type": "peer"}],
    )
    vnodes, vedges, from_store = hg._build_model(slug, thesis)

    assert from_store is False  # store had no edges → fall back to the file
    assert len(vedges) == 1 and vedges[0].confidence is None


# --- AppTest render ---------------------------------------------------------


def test_halo_graph_page_renders_without_exception():
    """Fallback path (empty tmp state.db) against the real repo theses."""
    at = AppTest.from_file(str(PAGES_DIR / "halo_graph.py"), default_timeout=DASHBOARD_TIMEOUT_S)
    at.run()
    assert not at.exception, f"page raised: {[e.message for e in at.exception]}"
    assert any(
        sb.label == "Thesis" for sb in at.selectbox
    ), f"thesis selector missing; saw: {[s.label for s in at.selectbox]}"


def test_halo_graph_page_renders_populated_graph():
    """Seed a graph for an on-disk thesis slug; the page defaults to it and
    labels the source as the discovery graph (rich path exercised)."""
    from data import state as state_db

    slug = thesis = None
    for s, _ in hg._list_theses():
        t = hg._load_thesis(s)
        if t and len(t.universe) >= 2:
            slug, thesis = s, t
            break
    assert slug, "repo needs a thesis with >=2 universe tickers"

    uni = thesis.universe[:3]
    now = state_db._now_iso()
    gnodes = [
        GraphNode(ticker=t, name=f"{t} Inc", thesis_slug=slug, first_seen=now, last_seen=now)
        for t in uni
    ]
    gedges = [
        GraphEdge(
            **{"from": uni[0]},
            to=uni[1],
            type="peer",
            note="seeded",
            confidence=0.6,
            grounded=True,
            evidence=[],
            thesis_slug=slug,
            as_of=now,
        )
    ]
    graph.save_graph(slug, gnodes, gedges)  # into the autouse tmp DB_PATH

    at = AppTest.from_file(str(PAGES_DIR / "halo_graph.py"), default_timeout=DASHBOARD_TIMEOUT_S)
    at.run()
    assert not at.exception, f"page raised: {[e.message for e in at.exception]}"
    blob = " ".join(m.value for m in at.markdown)
    assert "discovery graph" in blob, "populated graph should label its source as the store"
