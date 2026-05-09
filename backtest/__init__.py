"""Backtest harness (Step B3).

Replays the FINAQ drill-in graph against historical data ("as of" a fixed
date), captures the report's predictions (P10/P50/P90 fair values,
directional verdict, action recommendation), then compares against actual
realised prices at +30 / +90 / +180 days.

Public surface:
  - `backtest.runner.run_backtest(ticker, as_of_date, horizons)` — one full
    drill-in + score; returns a `BacktestResult` JSON-friendly dict.
  - `backtest.thesis_resolver.resolve_thesis(ticker, as_of_date)` — returns
    `(slug, thesis_dict)` for the (ticker, as_of) pair. Picks an existing
    curated thesis if the ticker is in its universe; otherwise generates a
    date-pinned adhoc thesis cached at `theses/backtest/`.
  - `backtest.scorer.score_run(state, as_of_date, horizons)` — pulls actual
    prices at each horizon and computes calibration metrics.

Outputs land at:
  - `data_cache/backtest/runs/{TICKER}__{as_of}.json` — per-run state + scores
  - `data_cache/backtest/aggregate__{date}.md` — markdown rollup table

Forward-leakage posture: the caller is responsible for ensuring every
`MODEL_*` env var resolves to a model whose training cutoff is ≤ as_of_date.
The runner does NOT verify this — pre-flight check is part of the demo prep.
"""

from __future__ import annotations
