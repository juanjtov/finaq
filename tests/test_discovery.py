"""Unit tests for the Phase 2 Discovery agent (`agents/discovery.py`).

The proposal LLM call is stubbed via the `_propose` seam; grounding retrieval
is stubbed on `disc.vectors` / `disc.finnhub`; company-name resolution via
`disc._company_name_for`. Disk writes are redirected to `tmp_path` and the
graph store to an explicit `db_path`.
"""

from __future__ import annotations

import json

import pytest

from agents import discovery as disc
from data import graph

_VALUATION = {
    "equity_risk_premium": 0.05,
    "erp_basis": "test",
    "terminal_growth_rate": 0.025,
    "terminal_growth_basis": "test",
    "discount_rate_floor": 0.07,
    "discount_rate_cap": 0.12,
}

_NAMES = {
    "NVDA": "NVIDIA Corporation",
    "VRT": "Vertiv Holdings Co",
    "CEG": "Constellation Energy",
    "FAKE": "Fake Co",
}


def _proposal() -> dict:
    """A proposal with one filing-backed edge (VRT), one news-backed edge
    (CEG), one hallucinated edge (FAKE, no evidence), one out-of-universe edge
    (MSFT), and a junk ticker in the universe."""
    return {
        "name": "Test halo",
        "summary": "A test halo graph.",
        "anchor_tickers": ["NVDA"],
        "universe": ["NVDA", "VRT", "CEG", "FAKE", "bad ticker!"],
        "relationships": [
            {"from": "NVDA", "to": "VRT", "type": "customer", "note": "racks use VRT cooling"},
            {"from": "NVDA", "to": "CEG", "type": "peer", "note": "CEG powers NVDA datacenters"},
            {"from": "NVDA", "to": "FAKE", "type": "supplier", "note": "hallucinated"},
            {"from": "NVDA", "to": "MSFT", "type": "customer", "note": "out of universe"},
        ],
        "valuation": _VALUATION,
        "material_thresholds": [],
    }


def _fake_query(ticker, question, k=8, **kwargs):
    # Only NVDA's filings corroborate the VRT edge.
    if ticker == "NVDA":
        return [
            {
                "text": "Our systems use VRT liquid cooling from Vertiv.",
                "metadata": {"accession": "acc1", "item_code": "1", "filed_date": "2026-03-01"},
                "score": 0.9,
            }
        ]
    return []


def _fake_news(ticker, company_name=None, **kwargs):
    # Only NVDA news co-mentions CEG.
    if ticker == "NVDA":
        return [
            {
                "title": "NVIDIA and Constellation Energy CEG expand deal",
                "content": "A multi-year agreement.",
                "url": "https://example.com/a",
                "source": "Yahoo",
                "published_date": "2026-08-01",
            }
        ]
    return []


@pytest.fixture
def stub(monkeypatch, tmp_path):
    monkeypatch.setattr(disc, "THESES_DIR", tmp_path / "theses")
    monkeypatch.setattr(disc, "_propose", lambda *, topic, ticker: (_proposal(), "raw"))
    monkeypatch.setattr(disc, "_company_name_for", lambda t: _NAMES.get(t, t))
    monkeypatch.setattr(disc.vectors, "query", _fake_query)
    monkeypatch.setattr(disc.vectors, "has_ticker", lambda t: True)
    monkeypatch.setattr(disc.finnhub, "search_news", _fake_news)
    return tmp_path


async def test_grounds_and_prunes(stub):
    db = stub / "state.db"
    res = await disc.discover(topic="ai power", db_path=db)

    assert res.error is None and res.thesis is not None
    # Filing-backed (VRT) + news-backed (CEG) survive; hallucinated (FAKE) and
    # out-of-universe (MSFT) do not.
    kept = {(r.from_, r.to) for r in res.thesis.relationships}
    assert kept == {("NVDA", "VRT"), ("NVDA", "CEG")}
    assert res.n_edges_grounded == 2
    # 3 proposed edges had both endpoints in-universe (the MSFT edge was skipped).
    assert res.n_edges_proposed == 3
    # The junk symbol was dropped from the universe; FAKE stays even though its
    # edge was pruned.
    assert "BAD TICKER!" in res.dropped_tickers
    assert "FAKE" in res.thesis.universe

    # Thesis written under the redirected theses dir.
    assert res.path.exists()
    assert res.path.parent == stub / "theses"

    # Graph persisted with all 3 edges; 2 grounded.
    assert len(graph.get_edges(res.slug, db_path=db)) == 3
    assert len(graph.get_edges(res.slug, grounded_only=True, db_path=db)) == 2
    # The VRT edge carries a filing citation; CEG a news citation.
    by_to = {e.to: e for e in graph.get_edges(res.slug, db_path=db)}
    assert by_to["VRT"].evidence[0].source == "edgar"
    assert by_to["CEG"].evidence[0].source == "Yahoo"
    assert by_to["FAKE"].grounded is False


async def test_confidence_reflects_evidence_strength(stub):
    db = stub / "state.db"
    res = await disc.discover(topic="ai power", db_path=db)
    by_to = {e.to: e for e in graph.get_edges(res.slug, db_path=db)}
    # Filing hit (0.5) outranks a lone news co-mention (0.3).
    assert by_to["VRT"].confidence == pytest.approx(disc.FILING_WEIGHT)
    assert by_to["CEG"].confidence == pytest.approx(disc.NEWS_WEIGHT)
    assert by_to["FAKE"].confidence == 0.0


async def test_vague_input_returns_error(monkeypatch, tmp_path):
    monkeypatch.setattr(disc, "THESES_DIR", tmp_path / "theses")
    monkeypatch.setattr(
        disc, "_propose", lambda *, topic, ticker: ({"error": "input too vague"}, "raw")
    )
    res = await disc.discover(topic="stuff")
    assert res.thesis is None
    assert "vague" in (res.error or "")
    assert not res.path.exists()


async def test_unparseable_proposal_returns_error(monkeypatch, tmp_path):
    monkeypatch.setattr(disc, "THESES_DIR", tmp_path / "theses")
    monkeypatch.setattr(disc, "_propose", lambda *, topic, ticker: ({}, "garbage"))
    res = await disc.discover(topic="stuff")
    assert res.thesis is None and res.error is not None


async def test_requires_exactly_one_input():
    assert (await disc.discover()).error is not None
    assert (await disc.discover(topic="x", ticker="Y")).error is not None


async def test_cache_hit_skips_llm(monkeypatch, tmp_path):
    theses = tmp_path / "theses"
    theses.mkdir()
    (theses / "adhoc_ai_power.json").write_text(
        json.dumps(
            {
                "name": "Cached",
                "summary": "s",
                "anchor_tickers": ["NVDA"],
                "universe": ["NVDA", "VRT"],
                "relationships": [],
                "material_thresholds": [],
            }
        )
    )
    monkeypatch.setattr(disc, "THESES_DIR", theses)

    def _boom(**kwargs):
        raise AssertionError("must not call the LLM on a cache hit")

    monkeypatch.setattr(disc, "_propose", _boom)
    res = await disc.discover(topic="ai power")
    assert res.cached is True and res.error is None
    assert res.thesis.name == "Cached"


async def test_force_refresh_regenerates(stub):
    theses = stub / "theses"
    theses.mkdir(parents=True, exist_ok=True)
    (theses / "adhoc_ai_power.json").write_text(
        json.dumps(
            {
                "name": "Old",
                "summary": "s",
                "anchor_tickers": ["NVDA"],
                "universe": ["NVDA"],
                "relationships": [],
                "material_thresholds": [],
            }
        )
    )
    res = await disc.discover(topic="ai power", db_path=stub / "s.db", force_refresh=True)
    assert res.cached is False
    assert res.thesis.name == "Test halo"


def test_mention_hit_rejects_stopword_and_short_tickers():
    # A common-word ticker (CAT) or single-letter ticker (A) must NOT be
    # grounded by ordinary prose that merely contains those letters.
    assert disc._mention_hit("The firm runs a strong CAT business.", "CAT", "") is False
    assert disc._mention_hit("It operates on a thin margin.", "A", "Agilent Technologies") is False
    # ...but the distinctive company-name token still grounds them.
    assert disc._mention_hit("Agilent posted record revenue.", "A", "Agilent Technologies") is True
    # A distinctive symbol (>=3 chars, not a stopword) still matches directly.
    assert disc._mention_hit("Our racks use VRT cooling.", "VRT", "Vertiv Holdings Co") is True


async def test_duplicate_pair_does_not_lose_graph(stub, monkeypatch):
    # The LLM proposing the same (from, to) twice must not violate the
    # UNIQUE(slug, from, to) constraint and roll back the whole graph write.
    def _dup(*, topic, ticker):
        return (
            {
                "name": "Dup",
                "summary": "s",
                "anchor_tickers": ["NVDA"],
                "universe": ["NVDA", "VRT"],
                "relationships": [
                    {"from": "NVDA", "to": "VRT", "type": "supplier", "note": "a"},
                    {"from": "NVDA", "to": "VRT", "type": "customer", "note": "b"},
                ],
                "valuation": _VALUATION,
                "material_thresholds": [],
            },
            "raw",
        )

    monkeypatch.setattr(disc, "_propose", _dup)
    db = stub / "state.db"
    res = await disc.discover(topic="dup", db_path=db)
    assert res.error is None
    assert res.n_edges_proposed == 1  # the duplicate pair was collapsed
    # Graph persisted intact (the pre-fix bug rolled back to 0 nodes + 0 edges).
    assert len(graph.get_nodes(res.slug, db_path=db)) == 2
    assert len(graph.get_edges(res.slug, db_path=db)) == 1


async def test_ticker_mode_forces_seed_into_universe(stub, monkeypatch):
    # Model omits the seed ticker; discover must inject it and lead the anchors.
    def _no_seed(*, topic, ticker):
        return (
            {
                "name": "Halo",
                "summary": "s",
                "anchor_tickers": ["VRT"],
                "universe": ["VRT", "CEG"],
                "relationships": [],
                "valuation": _VALUATION,
                "material_thresholds": [],
            },
            "raw",
        )

    monkeypatch.setattr(disc, "_propose", _no_seed)
    res = await disc.discover(ticker="NVDA", db_path=stub / "s.db")
    assert res.error is None
    assert "NVDA" in res.thesis.universe
    assert res.thesis.anchor_tickers[0] == "NVDA"
