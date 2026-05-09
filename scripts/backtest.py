"""Backtest CLI — walks the (ticker × as_of × horizons) matrix and writes
per-run JSON + an aggregate markdown report.

Examples:

    # The demo matrix
    python -m scripts.backtest \\
        --tickers INTC,NU,COUR,WEN,CRM,NKE \\
        --as-of 2025-09-05 \\
        --horizons 30,90,180

    # Test any other ticker on demand
    python -m scripts.backtest \\
        --tickers SHOP \\
        --as-of 2025-09-05 \\
        --horizons 30,90,180

Each (ticker, as_of) lands in `data_cache/backtest/runs/{TICKER}__{as_of}.json`
with the full report state + scoring. The aggregate markdown rolls up
band coverage / direction accuracy / confidence calibration across all
runs for the chosen as_of. Runs are skipped if the JSON already exists
(use `--rerun` to force).

Outputs:
  - data_cache/backtest/runs/{TICKER}__{as_of}.json   (one per pair)
  - data_cache/backtest/aggregate__{as_of}.md         (rollup table)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Bootstrap so this can be `python -m scripts.backtest`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import logger

RUNS_DIR = Path("data_cache/backtest/runs")
AGG_DIR = Path("data_cache/backtest")


# --- Aggregate markdown report ---------------------------------------------


def _hit_rate(values: list[bool | None]) -> tuple[int, int, str]:
    """Returns (hits, n, percentage_str). Skips None."""
    scored = [v for v in values if v is not None]
    if not scored:
        return 0, 0, "—"
    hits = sum(1 for v in scored if v)
    return hits, len(scored), f"{hits}/{len(scored)} ({hits / len(scored) * 100:.0f}%)"


def _avg_pct_err(values: list[float | None]) -> str:
    """Mean of non-None abs pct errors, formatted as a percentage."""
    scored = [v for v in values if v is not None]
    if not scored:
        return "—"
    avg = sum(scored) / len(scored)
    return f"{avg * 100:.1f}%"


def write_aggregate(as_of_date: str, runs: list[dict]) -> Path:
    """Roll up per-run scores into a markdown table.

    Sections:
      1. Per-run summary (one row per ticker)
      2. Band coverage by horizon
      3. P50 magnitude error by horizon
      4. Direction accuracy by horizon
      5. Confidence calibration (P50 error grouped by HIGH / MEDIUM / LOW)
    """
    out = [
        f"# Backtest aggregate — as_of {as_of_date}",
        "",
        f"_Generated from {len(runs)} per-run JSON files in `data_cache/backtest/runs/`._",
        "",
        "## Per-run summary",
        "",
        "| Ticker | Thesis | Verdict | Confidence | Risk | P10 | P50 | P90 | as_of close | conv |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(runs, key=lambda r: r.get("ticker", "")):
        score = r.get("score") or {}
        mc = score.get("mc") or {}
        prices = score.get("prices") or {}
        as_of = (prices.get("as_of") or {}).get("close")
        out.append(
            f"| {r.get('ticker', '?')} "
            f"| `{r.get('thesis_slug', '?')}` "
            f"| {score.get('verdict', '—')} "
            f"| {score.get('synthesis_confidence') or '—'} "
            f"| {score.get('risk_level') or '—'} "
            f"| {_fmt_dollar(mc.get('p10'))} "
            f"| {_fmt_dollar(mc.get('p50'))} "
            f"| {_fmt_dollar(mc.get('p90'))} "
            f"| {_fmt_dollar(as_of)} "
            f"| {_fmt_float(mc.get('convergence_ratio'))} "
            f"|"
        )

    horizons = sorted({h for r in runs for h in (r.get("horizons") or [])})

    # --- Band coverage
    out += ["", "## Band coverage", "",
            "| Band | " + " | ".join(f"{h}d" for h in horizons) + " |",
            "|---|" + "|".join("---" for _ in horizons) + "|"]
    for band, key in (("P10–P90", "in_p10_p90"), ("P25–P75", "in_p25_p75")):
        cells = [band]
        for h in horizons:
            vals = [
                ((r.get("score") or {}).get("horizons") or {}).get(f"h_{h}", {}).get(key)
                for r in runs
            ]
            cells.append(_hit_rate(vals)[2])
        out.append("| " + " | ".join(cells) + " |")

    # --- P50 magnitude error
    out += ["", "## P50 magnitude error (mean abs pct error vs realised)", "",
            "| | " + " | ".join(f"{h}d" for h in horizons) + " |",
            "|---|" + "|".join("---" for _ in horizons) + "|"]
    cells = ["all"]
    for h in horizons:
        vals = [
            ((r.get("score") or {}).get("horizons") or {}).get(f"h_{h}", {}).get("abs_pct_err_vs_p50")
            for r in runs
        ]
        cells.append(_avg_pct_err(vals))
    out.append("| " + " | ".join(cells) + " |")

    # --- Direction accuracy
    out += ["", "## Direction accuracy (verdict matched realised move)", "",
            "| | " + " | ".join(f"{h}d" for h in horizons) + " |",
            "|---|" + "|".join("---" for _ in horizons) + "|"]
    cells = ["all"]
    for h in horizons:
        vals = [
            ((r.get("score") or {}).get("horizons") or {}).get(f"h_{h}", {}).get("direction_match")
            for r in runs
        ]
        cells.append(_hit_rate(vals)[2])
    out.append("| " + " | ".join(cells) + " |")

    # --- Confidence calibration
    out += ["", "## Confidence calibration (P50 abs pct error by confidence level)", "",
            "| Confidence | " + " | ".join(f"{h}d" for h in horizons) + " | n |",
            "|---|" + "|".join("---" for _ in horizons) + "|---|"]
    by_conf: dict[str, list[dict]] = {"high": [], "medium": [], "low": []}
    for r in runs:
        c = (r.get("synthesis_confidence") or "").lower()
        if c in by_conf:
            by_conf[c].append(r)
    for level in ("high", "medium", "low"):
        bucket = by_conf[level]
        if not bucket:
            continue
        cells = [level]
        for h in horizons:
            vals = [
                ((r.get("score") or {}).get("horizons") or {}).get(f"h_{h}", {}).get("abs_pct_err_vs_p50")
                for r in bucket
            ]
            cells.append(_avg_pct_err(vals))
        cells.append(str(len(bucket)))
        out.append("| " + " | ".join(cells) + " |")

    # --- Verdict mix sanity
    verdict_counts = Counter((r.get("score") or {}).get("verdict", "unknown") for r in runs)
    out += ["", "## Verdict mix",
            "",
            "| verdict | count |",
            "|---|---|"]
    for v, n in verdict_counts.most_common():
        out.append(f"| {v} | {n} |")

    AGG_DIR.mkdir(parents=True, exist_ok=True)
    path = AGG_DIR / f"aggregate__{as_of_date}.md"
    path.write_text("\n".join(out) + "\n")
    return path


def _fmt_dollar(v: object) -> str:
    try:
        return f"${float(v):,.2f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_float(v: object, decimals: int = 2) -> str:
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


# --- CLI -------------------------------------------------------------------


async def _run_matrix(
    tickers: list[str],
    as_of_date: str,
    horizons: list[int],
    *,
    rerun: bool,
    force_refresh_adhoc: bool,
) -> list[dict]:
    """Run all (ticker, as_of) pairs sequentially. Skip pairs whose JSON
    already exists unless `--rerun`."""
    from backtest.runner import run_backtest

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for ticker in tickers:
        ticker = ticker.upper()
        path = RUNS_DIR / f"{ticker}__{as_of_date}.json"
        if path.exists() and not rerun:
            try:
                results.append(json.loads(path.read_text()))
                logger.info(f"[backtest-cli] {ticker}: cached at {path}, skipping (use --rerun to force)")
                continue
            except Exception as e:
                logger.warning(f"[backtest-cli] {ticker}: cache read failed: {e}; rerunning")

        try:
            record = await run_backtest(
                ticker, as_of_date=as_of_date, horizons=horizons,
                force_refresh_adhoc=force_refresh_adhoc,
            )
            results.append(record)
        except Exception as e:
            logger.error(f"[backtest-cli] {ticker} as_of={as_of_date} FAILED: {e}", exc_info=True)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tickers", required=True,
        help="Comma-separated tickers (e.g. INTC,NU,COUR,WEN,CRM,NKE).",
    )
    parser.add_argument(
        "--as-of", required=True, dest="as_of_date",
        help="ISO date (YYYY-MM-DD). Must be ≤ every model's training cutoff.",
    )
    parser.add_argument(
        "--horizons", default="30,90,180",
        help="Comma-separated horizon days (default 30,90,180).",
    )
    parser.add_argument(
        "--rerun", action="store_true",
        help="Re-run pairs whose JSON already exists (default: skip).",
    )
    parser.add_argument(
        "--force-refresh-adhoc", action="store_true",
        help="Re-synthesize date-pinned adhoc theses (default: cache hit).",
    )
    args = parser.parse_args(argv)

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]

    logger.info(
        f"[backtest-cli] Running {len(tickers)} ticker(s) × as_of={args.as_of_date} "
        f"× horizons={horizons}"
    )
    results = asyncio.run(_run_matrix(
        tickers, args.as_of_date, horizons,
        rerun=args.rerun,
        force_refresh_adhoc=args.force_refresh_adhoc,
    ))
    if not results:
        logger.error("[backtest-cli] no successful runs — skipping aggregate")
        return 1

    agg_path = write_aggregate(args.as_of_date, results)
    logger.info(f"[backtest-cli] aggregate written → {agg_path}")
    print(f"\n=== Backtest complete: {len(results)} runs ===")
    print(f"Per-run JSONs: {RUNS_DIR}/")
    print(f"Aggregate:     {agg_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
