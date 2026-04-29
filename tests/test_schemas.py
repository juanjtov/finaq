"""Step 3 tests — Pydantic schemas + thesis JSON validation.

Pure-logic, no network. Run via:  pytest tests/test_schemas.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from utils.schemas import (
    OPERATORS,
    Evidence,
    FundamentalsOutput,
    MaterialThreshold,
    NewsItem,
    NewsOutput,
    Projections,
    Relationship,
    RiskOutput,
    Thesis,
    TopRisk,
)

THESES_DIR = Path(__file__).parents[1] / "theses"
EXPECTED_THESES = ("ai_cake", "nvda_halo", "construction")


# --- Each committed thesis JSON validates ------------------------------------


@pytest.mark.parametrize("slug", EXPECTED_THESES)
def test_thesis_json_validates(slug):
    path = THESES_DIR / f"{slug}.json"
    assert path.exists(), f"missing thesis file: {path}"
    thesis = Thesis.model_validate_json(path.read_text())
    assert thesis.universe, f"{slug}: universe is empty"
    assert thesis.material_thresholds, f"{slug}: no material thresholds"


# --- Universe / anchor / relationship invariants for every thesis ------------


@pytest.mark.parametrize("slug", EXPECTED_THESES)
def test_anchors_subset_of_universe(slug):
    thesis = Thesis.model_validate_json((THESES_DIR / f"{slug}.json").read_text())
    assert set(thesis.anchor_tickers).issubset(set(thesis.universe))


@pytest.mark.parametrize("slug", EXPECTED_THESES)
def test_all_relationships_reference_universe_tickers(slug):
    thesis = Thesis.model_validate_json((THESES_DIR / f"{slug}.json").read_text())
    universe = set(thesis.universe)
    for rel in thesis.relationships:
        assert rel.from_ in universe, f"{slug}: relationship.from {rel.from_!r} not in universe"
        assert rel.to in universe, f"{slug}: relationship.to {rel.to!r} not in universe"


@pytest.mark.parametrize("slug", EXPECTED_THESES)
def test_all_thresholds_use_allowed_operators(slug):
    thesis = Thesis.model_validate_json((THESES_DIR / f"{slug}.json").read_text())
    for th in thesis.material_thresholds:
        assert th.operator in OPERATORS, f"{slug}: bad operator {th.operator!r}"


# --- Negative cases ----------------------------------------------------------


def test_thesis_missing_anchor_tickers_raises():
    raw = {
        "name": "x",
        "summary": "y",
        "universe": ["NVDA"],
        "relationships": [],
        "material_thresholds": [],
    }
    with pytest.raises(ValidationError):
        Thesis.model_validate(raw)


def test_thesis_anchor_outside_universe_raises():
    raw = {
        "name": "x",
        "summary": "y",
        "anchor_tickers": ["NVDA"],
        "universe": ["MSFT"],  # NVDA not present
        "relationships": [],
        "material_thresholds": [],
    }
    with pytest.raises(ValidationError, match="anchor_tickers not in universe"):
        Thesis.model_validate(raw)


def test_thesis_relationship_outside_universe_raises():
    raw = {
        "name": "x",
        "summary": "y",
        "anchor_tickers": ["NVDA"],
        "universe": ["NVDA"],
        "relationships": [{"from": "NVDA", "to": "MSFT", "type": "peer"}],
        "material_thresholds": [],
    }
    with pytest.raises(ValidationError, match="not in universe"):
        Thesis.model_validate(raw)


def test_threshold_contains_requires_string_value():
    with pytest.raises(ValidationError, match="contains"):
        MaterialThreshold(signal="x", operator="contains", value=42)


def test_threshold_numeric_op_rejects_string_value():
    with pytest.raises(ValidationError, match=">"):
        MaterialThreshold(signal="x", operator=">", value="not a number")


def test_relationship_accepts_alias_from():
    rel = Relationship.model_validate({"from": "NVDA", "to": "MSFT", "type": "customer"})
    assert rel.from_ == "NVDA"
    assert rel.to == "MSFT"


# --- Agent output models compile + validate happy paths ----------------------


def test_fundamentals_output_validates_minimal_payload():
    payload = {
        "summary": "ok",
        "kpis": {"revenue_5y_cagr": 0.45},
        "projections": {
            "revenue_growth_mean": 0.20,
            "revenue_growth_std": 0.05,
            "margin_mean": 0.65,
            "margin_std": 0.04,
            "exit_multiple_mean": 28,
            "exit_multiple_std": 4,
        },
    }
    out = FundamentalsOutput.model_validate(payload)
    assert isinstance(out.projections, Projections)
    assert out.projections.exit_multiple_mean == 28


def test_news_output_rejects_unknown_sentiment():
    bad = {
        "summary": "x",
        "catalysts": [{"title": "t", "summary": "s", "sentiment": "spicy", "url": "https://x"}],
    }
    with pytest.raises(ValidationError):
        NewsOutput.model_validate(bad)


def test_top_risk_severity_clamped_to_1_through_5():
    with pytest.raises(ValidationError):
        TopRisk(title="x", severity=6, explanation="y")
    with pytest.raises(ValidationError):
        TopRisk(title="x", severity=0, explanation="y")
    TopRisk(title="x", severity=3, explanation="y")  # ok


def test_risk_output_clamps_score_0_to_10():
    """`score_0_to_10` is now derived from `level`; the model validator
    enforces the canonical mapping. Mismatches raise."""
    with pytest.raises(ValidationError):
        # Out of 0..10 range
        RiskOutput(level="LOW", score_0_to_10=11, top_risks=[], summary="x")
    with pytest.raises(ValidationError):
        # level + score don't match canonical mapping
        RiskOutput(level="LOW", score_0_to_10=10, top_risks=[], summary="x")
    # ok — LOW maps to 2
    RiskOutput(level="LOW", score_0_to_10=2, top_risks=[], summary="x")
    # ok — CRITICAL maps to 10
    RiskOutput(level="CRITICAL", score_0_to_10=10, top_risks=[], summary="x")


def test_risk_level_to_score_mapping_is_canonical():
    from utils.schemas import RISK_LEVEL_TO_SCORE

    assert RISK_LEVEL_TO_SCORE == {
        "LOW": 2,
        "MODERATE": 4,
        "ELEVATED": 6,
        "HIGH": 8,
        "CRITICAL": 10,
    }


def test_risk_output_accepts_convergent_signals_and_threshold_breaches():
    from utils.schemas import ConvergentSignal, ThresholdBreach

    out = RiskOutput(
        level="ELEVATED",
        score_0_to_10=6,
        top_risks=[],
        convergent_signals=[
            ConvergentSignal(
                theme="supply concentration",
                sources=["fundamentals", "news"],
                explanation="Both Fundamentals and News surfaced supply-concentration risk.",
            )
        ],
        threshold_breaches=[
            ThresholdBreach(
                signal="fcf_yield",
                operator="<",
                threshold_value=4,
                observed_value=1.84,
                explanation="FCF yield well below the 4% margin-of-safety floor.",
                source="fundamentals",
            )
        ],
        summary="x",
    )
    assert out.level == "ELEVATED"
    assert out.threshold_breaches[0].observed_value == 1.84


# --- Cross-thesis sanity check (Phase 3 pattern detection foreshadowing) -----


def test_overlapping_tickers_match_finaq_context():
    """VRT, CEG, ANET, PWR should appear in 2+ theses (per FINAQ_Context.docx)."""
    theses = [
        Thesis.model_validate_json((THESES_DIR / f"{slug}.json").read_text())
        for slug in EXPECTED_THESES
    ]
    counts: dict[str, int] = {}
    for t in theses:
        for ticker in t.universe:
            counts[ticker] = counts.get(ticker, 0) + 1
    multi = {t for t, c in counts.items() if c >= 2}
    assert {"VRT", "CEG", "ANET", "PWR"}.issubset(
        multi
    ), f"expected VRT/CEG/ANET/PWR in 2+ theses, got multi={multi}"


def test_evidence_and_news_item_minimal_construction():
    e = Evidence(source="edgar", accession="0001", item="1A", excerpt="risk text")
    assert e.source == "edgar"
    n = NewsItem(title="t", summary="s", sentiment="bull", url="https://x")
    assert n.sentiment == "bull"


def test_evidence_accepts_optional_as_of_freshness_marker():
    """The new `as_of` field carries a date/datetime so downstream agents can
    weight stale evidence less."""
    e = Evidence(
        source="edgar",
        accession="0001045810-24-000023",
        item="1A",
        excerpt="capacity constrained",
        as_of="2024-02-21",
    )
    assert e.as_of == "2024-02-21"

    # Optional — schema must still accept evidence without it (e.g., for derived metrics).
    e2 = Evidence(source="yfinance", note="historical CAGR")
    assert e2.as_of is None
