"""Hand-curated golden set for RAG retrieval quality.

Each entry is a known-relevant query against the ingested NVDA corpus, with
substring fragments that should appear in at least one chunk in the top-K.
"Recall@K passes" if any expected substring is present in any top-K chunk.

These queries were drafted from the actual NVDA 10-K filed 2025-02-26
(accession 0001045810-25-000023). When a thesis or ticker corpus changes
substantially, the labels should be revisited — see docs/POSTPONED.md §2.

To extend: add tickers / queries here, ensure the relevant filings are
ingested, and the eval suite picks them up automatically.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldenQuery:
    """One labelled query for retrieval-quality evaluation.

    Fields:
      query: the question we'd ask the RAG system.
      ticker: which ticker the chunks should come from.
      item_filter: optional Item code (e.g. "1A") to constrain the search.
      expected_substrings: any one in any top-K chunk is sufficient for recall@K.
                           Case-insensitive substring match.
      description: human-readable note explaining the test's intent.
    """

    query: str
    ticker: str
    item_filter: str | None
    expected_substrings: tuple[str, ...]
    description: str


GOLDEN_QUERIES: tuple[GoldenQuery, ...] = (
    GoldenQuery(
        query="Hopper architecture data-center demand",
        ticker="NVDA",
        item_filter="7",  # MD&A
        expected_substrings=("Hopper", "data center"),
        description="MD&A explicitly cites Hopper as the driver of Data Center revenue growth",
    ),
    GoldenQuery(
        query="AI Diffusion export control rule impact on semiconductor sales",
        ticker="NVDA",
        item_filter="1A",  # Risk Factors
        expected_substrings=("AI Diffusion", "export"),
        description="Risk Factors discusses the AI Diffusion export-control regulation",
    ),
    GoldenQuery(
        query="Mellanox networking technology integration",
        ticker="NVDA",
        item_filter=None,
        expected_substrings=("Mellanox",),
        description="Mellanox appears as a discriminative term — narrow recall test",
    ),
    GoldenQuery(
        query="Compute and Networking segment revenue growth",
        ticker="NVDA",
        item_filter="7",
        expected_substrings=("Compute & Networking", "Compute and Networking", "segment"),
        description="MD&A breaks revenue down by Compute & Networking vs Graphics segments",
    ),
    GoldenQuery(
        query="supply constraints Blackwell ramp",
        ticker="NVDA",
        item_filter=None,
        expected_substrings=("Blackwell", "supply", "constrained"),
        description="Tests retrieval of supply / Blackwell-related discussion across the filing",
    ),
    GoldenQuery(
        query="China and other restricted markets revenue exposure",
        ticker="NVDA",
        item_filter="1A",
        expected_substrings=("China", "restricted", "export"),
        description="Risk Factors discloses geographic concentration and export risks",
    ),
    GoldenQuery(
        query="research and development expenses fiscal year",
        ticker="NVDA",
        item_filter="7",
        expected_substrings=("research and development", "R&D"),
        description="MD&A discusses R&D expense growth — common but should be retrievable",
    ),
    GoldenQuery(
        query="customer concentration risk top customers",
        ticker="NVDA",
        item_filter="1A",
        expected_substrings=("customer", "concentration"),
        description="Risk Factors flags revenue concentration among large hyperscalers",
    ),
)


def chunks_match_golden(chunks: list[dict], gq: GoldenQuery) -> tuple[bool, str | None, int | None]:
    """Recall@K check: does any chunk in `chunks` contain any expected substring?

    Returns (passed, matching_substring, rank_of_first_match_1_indexed).
    """
    for rank, chunk in enumerate(chunks, start=1):
        text = (chunk.get("text") or "").lower()
        for needle in gq.expected_substrings:
            if needle.lower() in text:
                return True, needle, rank
    return False, None, None
