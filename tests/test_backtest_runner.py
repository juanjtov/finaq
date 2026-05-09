"""Tests for the backtest package (Step B3).

Cover thesis resolution, scorer math, and runner orchestration. Real
LLM / yfinance / EDGAR calls are stubbed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# --- thesis_resolver --------------------------------------------------------


def test_thesis_resolver_picks_curated_when_ticker_in_universe(tmp_path, monkeypatch):
    """CRM is in `saas_universe.universe` — resolver returns that thesis
    verbatim, no adhoc generation."""
    from backtest import thesis_resolver

    fake_theses = tmp_path / "theses"
    fake_theses.mkdir()
    (fake_theses / "saas_universe.json").write_text(json.dumps({
        "name": "SaaS Universe",
        "summary": "Cloud SaaS thesis.",
        "anchor_tickers": ["CRM"],
        "universe": ["CRM", "SNOW", "DDOG"],
        "relationships": [],
        "material_thresholds": [],
    }))
    monkeypatch.setattr(thesis_resolver, "THESES_DIR", fake_theses)

    import asyncio
    slug, thesis = asyncio.run(
        thesis_resolver.resolve_thesis("CRM", as_of_date="2025-09-05")
    )
    assert slug == "saas_universe"
    assert thesis["slug"] == "saas_universe"
    assert "CRM" in thesis["universe"]


def test_thesis_resolver_synthesizes_adhoc_when_no_curated_match(tmp_path, monkeypatch):
    """INTC isn't in any curated universe — resolver falls through to
    `synthesize_adhoc_thesis(ticker=..., as_of_date=...)`."""
    from backtest import thesis_resolver

    fake_theses = tmp_path / "theses"
    fake_theses.mkdir()
    (fake_theses / "ai_cake.json").write_text(json.dumps({
        "name": "AI cake",
        "summary": "...",
        "anchor_tickers": ["NVDA"],
        "universe": ["NVDA", "AVGO"],
        "relationships": [],
        "material_thresholds": [],
    }))
    monkeypatch.setattr(thesis_resolver, "THESES_DIR", fake_theses)

    captured: dict = {}
    from agents import adhoc_thesis as _adhoc

    class _ThesisStub:
        def model_dump_json(self):
            return json.dumps({
                "name": "INTC ad-hoc",
                "summary": "INTC framework",
                "anchor_tickers": ["INTC"],
                "universe": ["INTC"],
                "relationships": [],
                "material_thresholds": [],
            })

    class _FakeResult:
        slug = "adhoc_intc"
        error = None
        notion_url = None
        cached = False
        path = None
        thesis = _ThesisStub()

    async def _stub_synth(*, ticker=None, topic=None, as_of_date=None,
                          force_refresh=False):
        captured["ticker"] = ticker
        captured["as_of_date"] = as_of_date
        return _FakeResult()

    monkeypatch.setattr(_adhoc, "synthesize_adhoc_thesis", _stub_synth)

    import asyncio
    slug, thesis = asyncio.run(
        thesis_resolver.resolve_thesis("INTC", as_of_date="2025-09-05")
    )
    assert captured["ticker"] == "INTC"
    assert captured["as_of_date"] == "2025-09-05"
    assert slug == "adhoc_intc"
    assert thesis["slug"] == "adhoc_intc"


def test_thesis_resolver_raises_on_synthesis_failure(tmp_path, monkeypatch):
    """If the adhoc synthesizer returns an error, the resolver raises so
    the runner doesn't proceed with a broken thesis."""
    from backtest import thesis_resolver
    from agents import adhoc_thesis as _adhoc

    fake_theses = tmp_path / "theses"
    fake_theses.mkdir()
    monkeypatch.setattr(thesis_resolver, "THESES_DIR", fake_theses)

    class _FakeFail:
        slug = "adhoc_xyz"
        thesis = None
        error = "LLM refused"
        path = None
        notion_url = None
        cached = False

    async def _stub_synth(**kw):
        return _FakeFail()

    monkeypatch.setattr(_adhoc, "synthesize_adhoc_thesis", _stub_synth)

    import asyncio
    with pytest.raises(RuntimeError, match="adhoc thesis synthesis failed"):
        asyncio.run(thesis_resolver.resolve_thesis("XYZ", as_of_date="2025-09-05"))


# --- scorer -----------------------------------------------------------------


def test_extract_verdict_undervalued_phrasing():
    from backtest.scorer import extract_verdict

    md = (
        "# INTC — semis update\n\n**Confidence:** medium\n\n"
        "## What this means\n"
        "Intel is meaningfully cheap relative to the math. Add on dip below $20."
    )
    assert extract_verdict(md) == "undervalued"


def test_extract_verdict_overvalued_phrasing():
    from backtest.scorer import extract_verdict

    md = (
        "# NKE — apparel update\n\n**Confidence:** low\n\n"
        "## What this means\n"
        "Nike looks expensive relative to growth. Trim 20% above $80."
    )
    assert extract_verdict(md) == "overvalued"


def test_extract_verdict_fairly_priced_phrasing():
    from backtest.scorer import extract_verdict
    md = "## What this means\nThe stock is fairly priced. No action recommended."
    assert extract_verdict(md) == "fairly_priced"


def test_extract_verdict_unknown_when_silent():
    from backtest.scorer import extract_verdict
    assert extract_verdict("") == "unknown"
    assert extract_verdict("Some prose about the company") == "unknown"


def test_band_coverage_p10_p90_inside():
    from backtest.scorer import _band_coverage
    assert _band_coverage(50, 30, 70) is True
    assert _band_coverage(50, 60, 70) is False
    assert _band_coverage(None, 30, 70) is None
    assert _band_coverage(50, None, 70) is None


def test_signed_pct_error_handles_nones():
    from backtest.scorer import _abs_pct_error, _signed_pct_error
    assert _signed_pct_error(None, 100) is None
    assert _signed_pct_error(110, None) is None
    assert _signed_pct_error(110, 100) == pytest.approx(0.10)
    assert _signed_pct_error(90, 100) == pytest.approx(-0.10)
    assert _abs_pct_error(110, 100) == pytest.approx(0.10)
    assert _abs_pct_error(90, 100) == pytest.approx(0.10)


def test_direction_match_correct_calls():
    from backtest.scorer import _direction_match
    assert _direction_match(verdict="undervalued",
                            price_at_as_of=100, realised=110) is True
    assert _direction_match(verdict="undervalued",
                            price_at_as_of=100, realised=90) is False
    assert _direction_match(verdict="overvalued",
                            price_at_as_of=100, realised=90) is True
    assert _direction_match(verdict="overvalued",
                            price_at_as_of=100, realised=110) is False
    assert _direction_match(verdict="fairly_priced",
                            price_at_as_of=100, realised=104) is True
    assert _direction_match(verdict="fairly_priced",
                            price_at_as_of=100, realised=120) is False
    assert _direction_match(verdict="unknown",
                            price_at_as_of=100, realised=110) is None
    assert _direction_match(verdict="undervalued",
                            price_at_as_of=None, realised=110) is None


def test_score_run_assembles_per_horizon_metrics(monkeypatch):
    from backtest import scorer

    def _stub_realised(ticker, *, as_of_date, horizons):
        return {
            "as_of": {"date": as_of_date, "close": 100.0},
            "h_30": {"date": "2025-10-05", "close": 110.0},
            "h_90": {"date": "2025-12-04", "close": 95.0},
            "h_180": {"date": "2026-03-04", "close": 130.0},
        }

    monkeypatch.setattr(scorer, "realised_prices", _stub_realised)

    fake_state = {
        "monte_carlo": {
            "dcf": {"p10": 80, "p25": 95, "p50": 120, "p75": 140, "p90": 160},
            "current_price": 100.0,
            "convergence_ratio": 0.75,
        },
        "synthesis_confidence": "medium",
        "risk": {"level": "ELEVATED"},
        "report": (
            "## What this means\nThe stock is meaningfully cheap relative to the math; "
            "add on a dip."
        ),
    }
    out = scorer.score_run(
        ticker="INTC",
        as_of_date="2025-09-05",
        horizons=[30, 90, 180],
        state=fake_state,
    )
    assert out["ticker"] == "INTC"
    assert out["verdict"] == "undervalued"
    assert out["horizons"]["h_30"]["in_p10_p90"] is True
    assert out["horizons"]["h_30"]["in_p25_p75"] is True
    assert out["horizons"]["h_30"]["direction_match"] is True
    assert out["horizons"]["h_90"]["direction_match"] is False
    assert out["horizons"]["h_30"]["abs_pct_err_vs_p50"] == pytest.approx(10 / 120)


# --- aggregate report ------------------------------------------------------


def test_aggregate_writes_markdown_with_expected_sections(tmp_path, monkeypatch):
    from scripts import backtest as cli

    monkeypatch.setattr(cli, "AGG_DIR", tmp_path)

    runs = [
        {
            "ticker": "INTC", "thesis_slug": "adhoc_intc", "horizons": [30, 90, 180],
            "synthesis_confidence": "medium",
            "score": {
                "ticker": "INTC", "verdict": "undervalued",
                "synthesis_confidence": "medium", "risk_level": "ELEVATED",
                "mc": {"p10": 18, "p25": 20, "p50": 24, "p75": 28, "p90": 32,
                       "current_price": 21.0, "convergence_ratio": 0.6},
                "prices": {"as_of": {"close": 21.0}},
                "horizons": {
                    "h_30": {"in_p10_p90": True, "in_p25_p75": True,
                             "abs_pct_err_vs_p50": 0.10, "direction_match": True},
                    "h_90": {"in_p10_p90": True, "in_p25_p75": False,
                             "abs_pct_err_vs_p50": 0.20, "direction_match": False},
                    "h_180": {"in_p10_p90": False, "in_p25_p75": False,
                              "abs_pct_err_vs_p50": 0.40, "direction_match": True},
                },
            },
        },
        {
            "ticker": "CRM", "thesis_slug": "saas_universe", "horizons": [30, 90, 180],
            "synthesis_confidence": "high",
            "score": {
                "ticker": "CRM", "verdict": "fairly_priced",
                "synthesis_confidence": "high", "risk_level": "MODERATE",
                "mc": {"p10": 200, "p25": 230, "p50": 260, "p75": 290, "p90": 320,
                       "current_price": 245.0, "convergence_ratio": 0.85},
                "prices": {"as_of": {"close": 245.0}},
                "horizons": {
                    "h_30": {"in_p10_p90": True, "in_p25_p75": True,
                             "abs_pct_err_vs_p50": 0.04, "direction_match": True},
                    "h_90": {"in_p10_p90": True, "in_p25_p75": True,
                             "abs_pct_err_vs_p50": 0.08, "direction_match": True},
                    "h_180": {"in_p10_p90": True, "in_p25_p75": False,
                              "abs_pct_err_vs_p50": 0.15, "direction_match": False},
                },
            },
        },
    ]
    path = cli.write_aggregate("2025-09-05", runs)
    md = path.read_text()
    assert "# Backtest aggregate — as_of 2025-09-05" in md
    assert "## Per-run summary" in md
    assert "## Band coverage" in md
    assert "## Direction accuracy" in md
    assert "## Confidence calibration" in md
    assert "INTC" in md
    assert "CRM" in md
