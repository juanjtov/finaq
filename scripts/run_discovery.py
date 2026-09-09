"""CLI for the Phase 2 Discovery agent (ARCHITECTURE §13).

Propose-then-verify a halo-graph thesis from a free-text topic or a seed
ticker, print a summary of what was kept vs pruned, and persist the graph to
`state.db` + the grounded thesis to `theses/adhoc_{slug}.json`.

    python -m scripts.run_discovery --topic "AI datacenter power"
    python -m scripts.run_discovery --ticker NVDA
    python -m scripts.run_discovery --topic "defense semis" --force
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from agents.discovery import discover


async def _main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.run_discovery",
        description="Discover a grounded halo-graph thesis (Phase 2).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--topic", help="Free-text topic, e.g. 'AI datacenter power'.")
    group.add_argument("--ticker", help="Seed ticker to build a halo around, e.g. NVDA.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if a thesis for this input is already cached on disk.",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Download + embed SEC filings for un-ingested universe tickers before "
        "grounding (slower + embedding cost, but far stronger first-party evidence).",
    )
    args = parser.parse_args()

    result = await discover(
        topic=args.topic, ticker=args.ticker, force_refresh=args.force, ingest=args.ingest
    )
    if result.error or result.thesis is None:
        print(f"discovery failed: {result.error}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "slug": result.slug,
                "thesis_path": str(result.path),
                "cached": result.cached,
                "name": result.thesis.name,
                "anchor_tickers": result.thesis.anchor_tickers,
                "universe": result.thesis.universe,
                "n_universe": result.n_universe,
                "n_edges_proposed": result.n_edges_proposed,
                "n_edges_grounded": result.n_edges_grounded,
                "dropped_tickers": result.dropped_tickers,
                "relationships": [
                    {
                        "from": r.from_,
                        "to": r.to,
                        "type": r.type,
                        "note": r.note,
                    }
                    for r in result.thesis.relationships
                ],
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
