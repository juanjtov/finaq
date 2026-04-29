"""Pydantic models for thesis JSON, agent outputs, and Monte Carlo inputs.

The Thesis model (and its sub-models Relationship, MaterialThreshold) validate
the hand-written JSONs in /theses/. The agent-output models match the contracts
in CLAUDE.md §9 and are what each agent must return from `run()`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# --- Thesis ------------------------------------------------------------------

OPERATORS = (">", "<", "abs >", "contains")
Operator = Literal[">", "<", "abs >", "contains"]
RelationshipType = Literal["supplier", "customer", "peer", "competitor"]
NewsSentiment = Literal["bull", "bear", "neutral"]
RiskLevel = Literal["LOW", "MODERATE", "ELEVATED", "HIGH", "CRITICAL"]

# Risk level → 0-10 score mapping. The LLM emits `level` (categorical, model-stable);
# `score_0_to_10` is derived. Keeps spec-compliance with CLAUDE.md §9.4 while
# avoiding the unstable-numeric-scale problem documented in ARCHITECTURE §7.6e.
RISK_LEVEL_TO_SCORE: dict[str, int] = {
    "LOW": 2,
    "MODERATE": 4,
    "ELEVATED": 6,
    "HIGH": 8,
    "CRITICAL": 10,
}


class Relationship(BaseModel):
    """A directed link between two tickers in a thesis universe."""

    model_config = ConfigDict(populate_by_name=True)

    from_: str = Field(alias="from")
    to: str
    type: RelationshipType
    note: str = ""


class MaterialThreshold(BaseModel):
    """A single rule that turns raw signals into a material alert."""

    signal: str
    operator: Operator
    value: float | str
    unit: str = ""

    @model_validator(mode="after")
    def _value_type_matches_operator(self) -> MaterialThreshold:
        if self.operator == "contains" and not isinstance(self.value, str):
            raise ValueError("operator 'contains' requires a string value")
        if self.operator != "contains" and not isinstance(self.value, (int, float)):
            raise ValueError(f"operator {self.operator!r} requires a numeric value")
        return self


class Thesis(BaseModel):
    """The hand-written investment thesis the rest of the system runs against."""

    name: str
    summary: str
    anchor_tickers: list[str]
    universe: list[str]
    relationships: list[Relationship] = []
    material_thresholds: list[MaterialThreshold] = []

    @model_validator(mode="after")
    def _anchors_subset_of_universe(self) -> Thesis:
        missing = set(self.anchor_tickers) - set(self.universe)
        if missing:
            raise ValueError(f"anchor_tickers not in universe: {sorted(missing)}")
        return self

    @model_validator(mode="after")
    def _relationships_reference_universe(self) -> Thesis:
        universe = set(self.universe)
        for rel in self.relationships:
            if rel.from_ not in universe:
                raise ValueError(f"relationship.from {rel.from_!r} not in universe")
            if rel.to not in universe:
                raise ValueError(f"relationship.to {rel.to!r} not in universe")
        return self


# --- Agent outputs (CLAUDE.md §9) --------------------------------------------


class Evidence(BaseModel):
    """A flexible citation usable by any agent (yfinance KPI, EDGAR chunk, news URL).

    `as_of` is a freshness marker: ISO date string of when the underlying data
    was captured (yfinance fetch time / SEC filed_date / news published_date).
    Downstream agents (Risk, Synthesis) use it to weight stale evidence less.
    """

    source: str  # "yfinance" | "edgar" | "tavily" | "chromadb" | ...
    accession: str | None = None
    item: str | None = None
    url: str | None = None
    excerpt: str | None = None
    note: str | None = None
    as_of: str | None = None  # ISO date or datetime (e.g. "2026-04-27" or "2026-04-27T18:30:00Z")


class Projections(BaseModel):
    """Inputs to the Monte Carlo engine, emitted by the Fundamentals agent."""

    revenue_growth_mean: float
    revenue_growth_std: float
    margin_mean: float
    margin_std: float
    exit_multiple_mean: float
    exit_multiple_std: float


class FundamentalsOutput(BaseModel):
    summary: str
    kpis: dict[str, Any]
    projections: Projections
    evidence: list[Evidence] = []
    errors: list[str] = []


class MdnaQuote(BaseModel):
    text: str
    accession: str
    item: str | None = None


class FilingsOutput(BaseModel):
    summary: str
    risk_themes: list[str]
    mdna_quotes: list[MdnaQuote] = []
    evidence: list[Evidence] = []
    errors: list[str] = []


class NewsItem(BaseModel):
    title: str
    summary: str
    sentiment: NewsSentiment
    url: str
    as_of: str | None = None  # ISO date — usually `published_date` from the news source


class NewsOutput(BaseModel):
    summary: str
    catalysts: list[NewsItem] = []
    concerns: list[NewsItem] = []
    evidence: list[Evidence] = []
    errors: list[str] = []


class TopRisk(BaseModel):
    title: str
    severity: int = Field(ge=1, le=5)
    explanation: str
    sources: list[str] = []  # which worker agents surfaced it: ["fundamentals", "news"]


class ConvergentSignal(BaseModel):
    """A risk theme that surfaced in 2+ source agents — strongest cross-modal signal."""

    theme: str  # short descriptor, e.g. "supply concentration"
    sources: list[str]  # ≥2 of: "fundamentals" | "filings" | "news"
    explanation: str


class ThresholdBreach(BaseModel):
    """A material_threshold from the thesis JSON that the worker outputs imply has fired."""

    signal: str  # threshold's signal name from the thesis JSON
    operator: Operator
    threshold_value: float | str
    observed_value: float | str | None = None  # may be None if signal is qualitative
    explanation: str
    source: str  # "fundamentals" | "filings" | "news" — which agent surfaced the breach


class RiskOutput(BaseModel):
    """Output of the Risk agent — synthesis-only, reads other workers' outputs.

    `level` is the primary categorical judgment (model-stable). `score_0_to_10`
    is derived from `level` via RISK_LEVEL_TO_SCORE for spec-compatibility and
    quick visualisation. The substantive output is `top_risks` plus the new
    structured fields (`convergent_signals`, `threshold_breaches`).
    """

    level: RiskLevel
    score_0_to_10: int = Field(ge=0, le=10)
    top_risks: list[TopRisk]
    convergent_signals: list[ConvergentSignal] = []
    threshold_breaches: list[ThresholdBreach] = []
    summary: str
    errors: list[str] = []

    @model_validator(mode="after")
    def _score_matches_level(self) -> RiskOutput:
        """Score must equal the canonical mapping for the given level."""
        expected = RISK_LEVEL_TO_SCORE[self.level]
        if self.score_0_to_10 != expected:
            raise ValueError(
                f"score_0_to_10 ({self.score_0_to_10}) does not match level {self.level!r} "
                f"(expected {expected} per RISK_LEVEL_TO_SCORE)"
            )
        return self


class SynthesisOutput(BaseModel):
    report: str  # markdown per CLAUDE.md §11
    errors: list[str] = []
