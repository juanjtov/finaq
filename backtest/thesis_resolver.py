"""Per-ticker thesis resolution for backtest mode.

Strategy:
  1. If `ticker` is in any *curated* (non-adhoc) thesis universe, use that
     thesis verbatim — it gives the drill-in a thesis-aware framing
     identical to production.
  2. Otherwise, synthesize a date-pinned adhoc thesis via
     `agents.adhoc_thesis.synthesize_adhoc_thesis(ticker=..., as_of_date=...)`.
     The thesis JSON gets cached at `theses/backtest/adhoc_{slug}__{as_of}.json`
     so subsequent runs of the same (ticker, as_of) pair don't re-pay the
     synthesizer LLM.

Output: `(slug, thesis_dict_with_slug_field)` — the dict has `slug` set so
`agents.invoke_with_telemetry` writes the right value to `graph_runs.thesis`.
"""

from __future__ import annotations

import json
from pathlib import Path

from utils import logger

THESES_DIR = Path("theses")
ADHOC_PREFIX = "adhoc_"


def _list_curated_slugs() -> list[str]:
    """Slugs of every curated (non-adhoc) thesis JSON, alphabetical."""
    if not THESES_DIR.exists():
        return []
    return sorted(
        p.stem
        for p in THESES_DIR.glob("*.json")
        if not p.stem.startswith(ADHOC_PREFIX)
    )


def _find_curated_match(ticker: str) -> tuple[str, dict] | None:
    """Return `(slug, thesis_dict)` for the FIRST curated thesis whose
    universe contains `ticker`, or None when no curated thesis claims it.

    Search order is alphabetical (matches `_list_curated_slugs`). If a
    ticker appears in multiple theses (e.g. CRM in `saas_universe`), the
    alphabetically-first thesis wins. The backtest doesn't try to be
    smart about this — for tickers with multiple lenses we'd want to
    drill against each separately, but Step B3 keeps it to one
    drill-in per ticker for demo simplicity.
    """
    ticker_u = ticker.upper()
    for slug in _list_curated_slugs():
        path = THESES_DIR / f"{slug}.json"
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            logger.warning(f"[thesis_resolver] could not read {path}: {e}")
            continue
        universe = {t.upper() for t in (data.get("universe") or [])}
        if ticker_u in universe:
            data["slug"] = slug
            return slug, data
    return None


async def resolve_thesis(
    ticker: str,
    *,
    as_of_date: str,
    force_refresh_adhoc: bool = False,
) -> tuple[str, dict]:
    """Pick a thesis for `(ticker, as_of_date)`.

    Returns `(slug, thesis_dict)`. The dict's `slug` field is set so
    downstream consumers (graph_runs telemetry, Streamlit cache keying)
    have a stable identifier.

    For curated-thesis matches, the slug is e.g. `saas_universe`. For
    adhoc generation the slug is e.g. `adhoc_intc`.
    """
    # 1. Curated thesis takes priority — gives the drill-in real
    #    thematic context the demo audience can recognise.
    curated = _find_curated_match(ticker)
    if curated is not None:
        slug, thesis = curated
        logger.info(f"[thesis_resolver] {ticker} → curated thesis {slug!r}")
        return slug, thesis

    # 2. Otherwise synthesize a date-pinned adhoc thesis.
    from agents.adhoc_thesis import synthesize_adhoc_thesis

    logger.info(
        f"[thesis_resolver] {ticker}: no curated match — synthesizing "
        f"adhoc thesis as_of={as_of_date}"
    )
    result = await synthesize_adhoc_thesis(
        ticker=ticker,
        as_of_date=as_of_date,
        force_refresh=force_refresh_adhoc,
    )
    if result.error or result.thesis is None:
        raise RuntimeError(
            f"adhoc thesis synthesis failed for {ticker} as_of={as_of_date}: "
            f"{result.error or 'no thesis returned'}"
        )

    thesis_dict = json.loads(result.thesis.model_dump_json())
    thesis_dict["slug"] = result.slug
    return result.slug, thesis_dict
