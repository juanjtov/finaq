"""Hand-curated golden set for the `synthesis_reports` ChromaDB corpus.

Each entry is a CIO-relevant question that should retrieve content from the
right past drill-in section. We test recall@K — at least one expected
substring (case-insensitive) must appear somewhere in the top-K chunks.

The corpus is populated by `scripts.index_existing_reports` from
`data_cache/demos/*.json` — every section of every demo report becomes a
ChromaDB chunk. This eval verifies the planner can find the right report
section to fold into its decision prompt.

Maintenance: when reports get rotated (older demos archived, newer ones
ingested), revisit `expected_substrings` so the labels still match real
content.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CIOReportsGoldenQuery:
    """One labelled query for synthesis_reports retrieval evaluation.

    Fields:
      query: the question the CIO planner would ask via `cio.rag.query_past_reports`.
      ticker: scope by ticker metadata pre-filter.
      thesis: optional thesis slug to further scope. None = match any thesis
              for that ticker.
      expected_substrings: any one (case-insensitive) appearing in any top-K
                           chunk's text suffices for recall@K.
      description: human-readable note on what the planner wants this for.
    """

    query: str
    ticker: str
    thesis: str | None
    expected_substrings: tuple[str, ...]
    description: str


CIO_REPORTS_GOLDEN_QUERIES: tuple[CIOReportsGoldenQuery, ...] = (
    # NVDA / ai_cake — biggest universe, expected to retrieve cleanly
    CIOReportsGoldenQuery(
        query="What does the bull case say about NVDA's AI revenue trajectory?",
        ticker="NVDA",
        thesis="ai_cake",
        expected_substrings=("data center", "compute", "growth", "demand"),
        description="NVDA bull case should surface data-center / compute language.",
    ),
    CIOReportsGoldenQuery(
        query="Top risks for NVDA",
        ticker="NVDA",
        thesis="ai_cake",
        expected_substrings=("risk", "concentrat", "regulat", "compet", "supply"),
        description="Top risks section should surface canonical NVDA risks.",
    ),
    CIOReportsGoldenQuery(
        query="Monte Carlo fair value NVDA P50",
        ticker="NVDA",
        thesis="ai_cake",
        expected_substrings=("p50", "p10", "p90", "fair value", "$"),
        description="MC section should surface percentile language and dollars.",
    ),

    # AAPL adhoc thesis
    CIOReportsGoldenQuery(
        query="What does AAPL's thesis bet on?",
        ticker="AAPL",
        thesis=None,  # any thesis matches
        expected_substrings=("services", "iPhone", "ecosystem", "installed base"),
        description="AAPL thesis statement should mention services / iPhone.",
    ),

    # DELL nvda_halo (AI server play)
    CIOReportsGoldenQuery(
        query="Action recommendation for DELL",
        ticker="DELL",
        thesis="nvda_halo",
        expected_substrings=("watch", "trim", "hold", "size", "position", "if"),
        description="Action rec should be a position-management instruction.",
    ),

    # Construction thesis — different vocabulary
    CIOReportsGoldenQuery(
        query="What's the watchlist for construction names?",
        ticker="CAT",
        thesis="construction",
        expected_substrings=("backlog", "earnings", "guidance", "infrastructure", "spending"),
        description="Construction watchlist should use sector-native language.",
    ),

    # Cross-section retrieval — "what changed" hits multiple sections
    CIOReportsGoldenQuery(
        query="data center capex outlook for hyperscalers",
        ticker="NVDA",
        thesis=None,
        expected_substrings=("capex", "data center", "hyperscaler", "compute"),
        description="Hyperscaler capex is a CIO-typical question.",
    ),
)
