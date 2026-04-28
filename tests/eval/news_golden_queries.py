"""Hand-curated golden set for News-agent retrieval & extraction quality.

Each entry asserts: "for ticker X, in the last N days, the News agent should
surface at least one catalyst or concern whose summary or title contains one
of the expected keywords."

Bar is recall@all_items (across catalysts + concerns combined), not @K. The
News agent typically returns 6–14 items per drill-in; we just want to know
whether the system *captured* the topic at all, not where it ranked.

To extend: add tuples here. When a thesis or ticker corpus changes, the
labels must be revisited — see docs/POSTPONED.md §2.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NewsGoldenQuery:
    """One labelled expectation for News-agent quality evaluation.

    Fields:
      ticker: the ticker the News agent runs against.
      thesis_slug: which thesis to use as the lens.
      expected_keywords: any one in any catalyst/concern's title or summary
                         is sufficient (case-insensitive).
      window_days: maximum article age (matches NEWS_DAYS).
      description: human-readable note explaining what we're checking.
    """

    ticker: str
    thesis_slug: str
    expected_keywords: tuple[str, ...]
    window_days: int
    description: str


# These should be true for NVDA at any time during 2026 — chip generations,
# China export controls, hyperscaler capex, and AI semiconductor demand are
# durable themes with multiple in-window articles.
NEWS_GOLDEN_QUERIES: tuple[NewsGoldenQuery, ...] = (
    NewsGoldenQuery(
        ticker="NVDA",
        thesis_slug="ai_cake",
        expected_keywords=("Blackwell", "Hopper", "Vera Rubin", "GB200", "Rubin"),
        window_days=90,
        description="A current-or-next-generation NVIDIA chip platform should be in the news",
    ),
    NewsGoldenQuery(
        ticker="NVDA",
        thesis_slug="ai_cake",
        expected_keywords=("China", "export", "license"),
        window_days=90,
        description="Export-control / China discussion is a durable AI-cake theme",
    ),
    NewsGoldenQuery(
        ticker="NVDA",
        thesis_slug="ai_cake",
        expected_keywords=("Microsoft", "Meta", "Google", "Amazon", "Oracle", "hyperscaler"),
        window_days=90,
        description="At least one hyperscaler partnership / capex story should be present",
    ),
    NewsGoldenQuery(
        ticker="NVDA",
        thesis_slug="ai_cake",
        expected_keywords=("revenue", "earnings", "guidance", "fiscal"),
        window_days=90,
        description="Recent earnings or guidance — always recurring",
    ),
    NewsGoldenQuery(
        ticker="NVDA",
        thesis_slug="ai_cake",
        expected_keywords=("data center", "AI infrastructure", "AI capex", "compute"),
        window_days=90,
        description="Core AI-cake thesis topic — data-center buildout and AI compute demand",
    ),
)


def items_match_golden(items: list[dict], gq: NewsGoldenQuery) -> tuple[bool, str | None]:
    """Recall check across catalysts + concerns combined.

    `items` is a list of dicts (each with at least `title` and `summary`).
    Returns (passed, matching_keyword).
    """
    haystack = " ".join(
        ((it.get("title") or "") + " " + (it.get("summary") or "")).lower() for it in items
    )
    for kw in gq.expected_keywords:
        if kw.lower() in haystack:
            return True, kw
    return False, None
