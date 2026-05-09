"""Backtest dashboard page (Step B4).

Read-only view of `data_cache/backtest/runs/*.json` + an optional
"test any ticker" form that fires `scripts.backtest` against an
arbitrary new ticker (writes a new per-run JSON, refreshes the page).

Sections:
  1. Aggregate calibration tables (band coverage, P50 magnitude error,
     direction accuracy, confidence calibration) across all runs of a
     selected as_of date.
  2. Per-run drill-down — pick a ticker, see its full report + scoring.
  3. "Test any ticker" form — input ticker(s), click Run, the runner
     fires in the same Python process and writes a new JSON.
"""

from __future__ import annotations

# Bootstrap so the page can be loaded standalone via `streamlit run`.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio
import json
from collections import Counter, defaultdict
from typing import Any

import pandas as pd
import streamlit as st

from ui.components import page_header, section_divider

st.set_page_config(page_title="FINAQ — Backtest", page_icon="📊", layout="wide")

RUNS_DIR = Path(__file__).parents[2] / "data_cache" / "backtest" / "runs"
AGG_DIR = Path(__file__).parents[2] / "data_cache" / "backtest"


# --- Data loading ----------------------------------------------------------


def _load_all_runs() -> list[dict]:
    """Read every per-run JSON from `data_cache/backtest/runs/`."""
    if not RUNS_DIR.exists():
        return []
    out: list[dict] = []
    for path in sorted(RUNS_DIR.glob("*.json")):
        try:
            out.append(json.loads(path.read_text()))
        except Exception as e:
            st.warning(f"Skipping unreadable run `{path.name}`: {e}")
    return out


def _runs_by_as_of(runs: list[dict]) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by[r.get("as_of_date", "?")].append(r)
    return by


# --- Aggregate metrics (mirror scripts/backtest.write_aggregate logic) ---


def _hit_rate(values: list[bool | None]) -> str:
    scored = [v for v in values if v is not None]
    if not scored:
        return "—"
    hits = sum(1 for v in scored if v)
    return f"{hits}/{len(scored)} ({hits / len(scored) * 100:.0f}%)"


def _avg_pct_err(values: list[float | None]) -> str:
    scored = [v for v in values if v is not None]
    if not scored:
        return "—"
    return f"{sum(scored) / len(scored) * 100:.1f}%"


def _fmt_dollar(v: object) -> str:
    try:
        return f"${float(v):,.2f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_pct(v: object) -> str:
    try:
        return f"{float(v) * 100:+.1f}%"
    except (TypeError, ValueError):
        return "—"


# --- Section renderers ----------------------------------------------------


def _render_per_run_summary(runs: list[dict]) -> None:
    rows: list[dict] = []
    for r in sorted(runs, key=lambda r: r.get("ticker", "")):
        score = r.get("score") or {}
        mc = score.get("mc") or {}
        prices = score.get("prices") or {}
        rows.append({
            "ticker": r.get("ticker", "?"),
            "thesis": r.get("thesis_slug", "?"),
            "verdict": score.get("verdict", "—"),
            "confidence": score.get("synthesis_confidence") or "—",
            "risk": score.get("risk_level") or "—",
            "P10": _fmt_dollar(mc.get("p10")),
            "P50": _fmt_dollar(mc.get("p50")),
            "P90": _fmt_dollar(mc.get("p90")),
            "as_of close": _fmt_dollar((prices.get("as_of") or {}).get("close")),
            "convergence": (
                f"{float(mc['convergence_ratio']):.2f}"
                if isinstance(mc.get("convergence_ratio"), (int, float))
                else "—"
            ),
            "duration_s": (
                f"{r['duration_s']:.0f}" if r.get("duration_s") else "—"
            ),
        })
    if not rows:
        st.caption("No runs yet for this as_of date.")
        return
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _render_calibration_tables(runs: list[dict]) -> None:
    horizons = sorted({h for r in runs for h in (r.get("horizons") or [])})
    if not horizons:
        st.caption("No horizons recorded — runs may have failed.")
        return

    # Band coverage
    st.markdown("**Band coverage** — did actual close fall inside the predicted band?")
    band_rows = []
    for label, key in (("P10–P90", "in_p10_p90"), ("P25–P75", "in_p25_p75")):
        row: dict = {"band": label}
        for h in horizons:
            vals = [
                ((r.get("score") or {}).get("horizons") or {}).get(f"h_{h}", {}).get(key)
                for r in runs
            ]
            row[f"{h}d"] = _hit_rate(vals)
        band_rows.append(row)
    st.dataframe(pd.DataFrame(band_rows), hide_index=True, width="stretch")

    # P50 error
    st.markdown("**P50 magnitude error** — mean absolute %-error vs realised close")
    err_row: dict = {"metric": "all runs"}
    for h in horizons:
        vals = [
            ((r.get("score") or {}).get("horizons") or {}).get(f"h_{h}", {}).get("abs_pct_err_vs_p50")
            for r in runs
        ]
        err_row[f"{h}d"] = _avg_pct_err(vals)
    st.dataframe(pd.DataFrame([err_row]), hide_index=True, width="stretch")

    # Direction accuracy
    st.markdown("**Direction accuracy** — did the verdict match the realised move?")
    dir_row: dict = {"metric": "all runs"}
    for h in horizons:
        vals = [
            ((r.get("score") or {}).get("horizons") or {}).get(f"h_{h}", {}).get("direction_match")
            for r in runs
        ]
        dir_row[f"{h}d"] = _hit_rate(vals)
    st.dataframe(pd.DataFrame([dir_row]), hide_index=True, width="stretch")

    # Confidence calibration
    st.markdown("**Confidence calibration** — P50 abs %-error grouped by confidence")
    by_conf: dict[str, list[dict]] = {"high": [], "medium": [], "low": []}
    for r in runs:
        c = (r.get("synthesis_confidence") or "").lower()
        if c in by_conf:
            by_conf[c].append(r)
    conf_rows = []
    for level in ("high", "medium", "low"):
        bucket = by_conf[level]
        if not bucket:
            continue
        row: dict = {"confidence": level, "n": len(bucket)}
        for h in horizons:
            vals = [
                ((r.get("score") or {}).get("horizons") or {}).get(f"h_{h}", {}).get("abs_pct_err_vs_p50")
                for r in bucket
            ]
            row[f"{h}d"] = _avg_pct_err(vals)
        conf_rows.append(row)
    if conf_rows:
        st.dataframe(pd.DataFrame(conf_rows), hide_index=True, width="stretch")
    else:
        st.caption("No confidence-tagged runs yet.")

    # Verdict mix
    st.markdown("**Verdict mix**")
    verdict_counts = Counter(
        (r.get("score") or {}).get("verdict", "unknown") for r in runs
    )
    if verdict_counts:
        st.dataframe(
            pd.DataFrame(
                [{"verdict": v, "count": n} for v, n in verdict_counts.most_common()]
            ),
            hide_index=True, width="stretch",
        )


def _render_drilldown(runs: list[dict]) -> None:
    """Per-ticker drill-down: pick a run, see its full state."""
    if not runs:
        return
    labels = [
        f"{r.get('ticker', '?')} — verdict={r.get('score', {}).get('verdict', '?')} "
        f"conf={r.get('synthesis_confidence') or '—'}"
        for r in runs
    ]
    chosen_label = st.selectbox("Pick a run", labels, index=0)
    run = runs[labels.index(chosen_label)]
    score = run.get("score") or {}

    cols = st.columns([1, 1, 1, 1, 1])
    mc = score.get("mc") or {}
    cols[0].metric("Verdict", score.get("verdict", "—"))
    cols[1].metric("Confidence", score.get("synthesis_confidence") or "—")
    cols[2].metric("Risk", score.get("risk_level") or "—")
    cols[3].metric("MC P50", _fmt_dollar(mc.get("p50")))
    cols[4].metric(
        "Convergence",
        f"{float(mc['convergence_ratio']):.2f}"
        if isinstance(mc.get("convergence_ratio"), (int, float))
        else "—",
    )

    # Side-by-side charts: MC histogram (with as-of price line) + time-series
    # (realised closes vs MC band). Both palette-aware via utils/charts.
    prices = score.get("prices") or {}
    as_of_close = (prices.get("as_of") or {}).get("close")
    horizons = run.get("horizons") or []
    realised_for_chart: list[tuple[int, str, float | None]] = [
        (h, prices.get(f"h_{h}", {}).get("date") or "", prices.get(f"h_{h}", {}).get("close"))
        for h in horizons
    ]

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.markdown("##### Monte Carlo distribution")
        try:
            from utils.charts import mc_histogram, resolve_mc_samples

            mc_with_samples = {
                "samples": None,
                "dcf": {
                    "p10": mc.get("p10"), "p25": mc.get("p25"), "p50": mc.get("p50"),
                    "p75": mc.get("p75"), "p90": mc.get("p90"),
                },
            }
            samples = resolve_mc_samples(mc_with_samples)
            if samples and as_of_close is not None:
                fig = mc_histogram(
                    samples,
                    current_price=float(as_of_close),
                    title=f"As-of {run.get('as_of_date', '')} fair-value distribution",
                )
                st.pyplot(fig, use_container_width=True)
                import matplotlib.pyplot as _plt

                _plt.close(fig)
                st.caption(
                    "Bars: simulated fair value per share. Dashed ink line: "
                    "actual close on the as_of date — where the market had "
                    "the stock vs the math."
                )
            else:
                st.caption("No MC distribution available for this run.")
        except Exception as e:
            st.warning(f"MC histogram render failed: {e}")

    with chart_cols[1]:
        st.markdown("##### Realised price vs MC bands")
        try:
            from utils.charts import backtest_price_path

            mc_pcts = {
                "p10": mc.get("p10"), "p25": mc.get("p25"), "p50": mc.get("p50"),
                "p75": mc.get("p75"), "p90": mc.get("p90"),
            }
            fig = backtest_price_path(
                as_of_date=run.get("as_of_date", ""),
                as_of_close=as_of_close,
                realised=realised_for_chart,
                mc_percentiles=mc_pcts,
                title="Realised closes (30/90/180d) vs MC fair-value bands",
            )
            st.pyplot(fig, use_container_width=True)
            import matplotlib.pyplot as _plt

            _plt.close(fig)
            st.caption(
                "Sage horizontal line = MC P50; parchment band = P10-P90; "
                "eggshell band = P25-P75. Sage circles = realised closes "
                "with %-return from as-of."
            )
        except Exception as e:
            st.warning(f"Time-series render failed: {e}")

    # Per-horizon panel
    st.markdown("##### Per-horizon scoring")
    horizon_rows = []
    for h in horizons:
        cell = (score.get("horizons") or {}).get(f"h_{h}", {})
        horizon_rows.append({
            "horizon": f"{h}d",
            "target_date": cell.get("target_date", "—"),
            "actual close": _fmt_dollar(cell.get("actual_close")),
            "realised return": _fmt_pct(cell.get("actual_return_from_as_of")),
            "in P10–P90": "✓" if cell.get("in_p10_p90") else (
                "✗" if cell.get("in_p10_p90") is False else "—"
            ),
            "in P25–P75": "✓" if cell.get("in_p25_p75") else (
                "✗" if cell.get("in_p25_p75") is False else "—"
            ),
            "|err vs P50|": (
                f"{cell['abs_pct_err_vs_p50'] * 100:.1f}%"
                if isinstance(cell.get("abs_pct_err_vs_p50"), (int, float))
                else "—"
            ),
            "direction": "✓" if cell.get("direction_match") else (
                "✗" if cell.get("direction_match") is False else "—"
            ),
        })
    if horizon_rows:
        st.dataframe(pd.DataFrame(horizon_rows), hide_index=True, width="stretch")

    # Errors (if any)
    errors = run.get("errors") or []
    if errors:
        st.warning(f"Run had {len(errors)} error(s):")
        for e in errors[:10]:
            st.code(str(e)[:300], language="text")

    # Report markdown
    with st.expander("Full synthesis report (markdown)"):
        st.markdown(run.get("report") or "_(empty)_")


def _render_run_form(default_as_of: str) -> None:
    """Form to fire `scripts.backtest` against an arbitrary ticker.

    Pressing Run kicks off the runner in the SAME Python process.
    Streamlit blocks during the run (~3-5 min/ticker). On completion the
    page reruns and the new JSON appears in the aggregate above.
    """
    st.markdown(
        "Enter one or more tickers and an as_of date. The runner will "
        "auto-resolve the thesis (curated if matched, else synthesises a "
        "date-pinned adhoc) and persist a per-run JSON to "
        "`data_cache/backtest/runs/`."
    )
    cols = st.columns([2, 1, 1, 1])
    with cols[0]:
        tickers_str = st.text_input(
            "Tickers (comma-separated)",
            value="",
            placeholder="e.g. SHOP or AAPL,MSFT",
            help="Single ticker or comma-separated list. Each one runs sequentially.",
        )
    with cols[1]:
        as_of = st.text_input(
            "as_of date (YYYY-MM-DD)",
            value=default_as_of,
            help="Must be ≤ every model's training cutoff to avoid forward leakage.",
        )
    with cols[2]:
        horizons_str = st.text_input(
            "Horizons (days)", value="30,90,180",
            help="Comma-separated. Realised closes pulled at as_of + each.",
        )
    with cols[3]:
        rerun = st.checkbox(
            "Rerun if cached", value=False,
            help="Skip cached pairs by default; check to force re-run.",
        )

    if st.button("▶ Run backtest", type="primary"):
        tickers = [t.strip().upper() for t in (tickers_str or "").split(",") if t.strip()]
        horizons = [int(h.strip()) for h in (horizons_str or "").split(",") if h.strip()]
        if not tickers:
            st.error("Enter at least one ticker.")
            return
        if not as_of:
            st.error("Enter an as_of date in YYYY-MM-DD format.")
            return

        from backtest.runner import run_backtest

        progress = st.progress(0.0, text=f"Starting {len(tickers)} run(s)…")
        for i, ticker in enumerate(tickers):
            existing = RUNS_DIR / f"{ticker}__{as_of}.json"
            if existing.exists() and not rerun:
                st.info(f"⏭ Skipping {ticker} — cached at `{existing.name}` (toggle Rerun to force).")
            else:
                progress.progress(
                    i / len(tickers),
                    text=f"Running {ticker} as_of={as_of}…",
                )
                try:
                    asyncio.run(run_backtest(
                        ticker, as_of_date=as_of, horizons=horizons,
                    ))
                    st.success(f"✓ {ticker} done.")
                except Exception as e:
                    st.error(f"✗ {ticker} failed: {e}")
            progress.progress(
                (i + 1) / len(tickers),
                text=f"{i + 1}/{len(tickers)} complete",
            )
        progress.empty()
        st.success("All runs complete. Refresh the page to see updated aggregates.")
        st.button("🔄 Refresh page", on_click=lambda: st.rerun())


# --- Page body ------------------------------------------------------------


def main() -> None:
    page_header(
        "Backtest",
        subtitle=(
            "Replay historical drill-ins as-of a fixed date, score against "
            "realised prices at +30 / +90 / +180 days."
        ),
    )

    runs = _load_all_runs()
    by_as_of = _runs_by_as_of(runs)

    if not runs:
        st.info(
            "No backtest runs on disk yet. Use the form below to fire one, "
            "or run from the CLI:\n\n"
            "```sh\n"
            "python -m scripts.backtest \\\n"
            "    --tickers INTC,NU,COUR,WEN,CRM,NKE \\\n"
            "    --as-of 2025-09-05 \\\n"
            "    --horizons 30,90,180\n"
            "```"
        )
        section_divider()
        st.subheader("Test any ticker")
        _render_run_form(default_as_of="2025-09-05")
        return

    # Sidebar: as_of date picker
    as_of_dates = sorted(by_as_of.keys(), reverse=True)
    chosen_as_of = st.sidebar.selectbox(
        "Backtest as_of date",
        as_of_dates,
        index=0,
    )
    selected = by_as_of[chosen_as_of]

    cols = st.columns(4)
    cols[0].metric("Runs in this batch", len(selected))
    horizons = sorted({h for r in selected for h in (r.get("horizons") or [])})
    cols[1].metric("Horizons covered", ", ".join(f"{h}d" for h in horizons) or "—")
    verdict_counts = Counter(
        (r.get("score") or {}).get("verdict", "unknown") for r in selected
    )
    cols[2].metric("Drills with verdict", sum(verdict_counts.values()) - verdict_counts.get("unknown", 0))
    cols[3].metric("As_of date", chosen_as_of)

    section_divider()
    st.subheader("Per-run summary")
    _render_per_run_summary(selected)

    section_divider()
    st.subheader("Calibration aggregate")
    st.caption(
        "How well-calibrated are the predictions across all runs at this as_of? "
        "Bigger numbers in the band-coverage and direction-accuracy rows are better; "
        "smaller numbers in the P50-error rows are better."
    )
    _render_calibration_tables(selected)

    section_divider()
    st.subheader("Per-ticker drill-down")
    _render_drilldown(selected)

    section_divider()
    st.subheader("Test any ticker")
    _render_run_form(default_as_of=chosen_as_of)


main()
