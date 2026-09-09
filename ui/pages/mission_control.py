"""Mission Control page — at-a-glance system health.

Reads from existing artefacts (no Step 5z dependency):
  - data_cache/eval/runs/*.json — eval run history (Tier 1 + Tier 2 + RAGAS)
  - data_cache/edgar/ — last-touched timestamps per ticker
  - data_cache/yfin/ — last-touched per ticker
  - data_cache/state.db ingested_filings — filings-index manifest (tickers, filings, chunks)
  - data_cache/demos/ — cached drill-in count

When Step 5z lands, this page will additionally read `data_cache/state.db`
for graph-run history, daily cost, and per-node telemetry. Until then,
those panels show "not yet recording" placeholders.
"""

from __future__ import annotations

# Bootstrap (see ui/app.py for explanation).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import os
from datetime import UTC, datetime

import pandas as pd
import streamlit as st

from ui.components import freshness_card, page_header, section_divider

st.set_page_config(page_title="FINAQ — Mission Control", page_icon="🛰️", layout="wide")

EVAL_DIR = Path(__file__).parents[2] / "data_cache" / "eval" / "runs"
DEMO_DIR = Path(__file__).parents[2] / "data_cache" / "demos"
EDGAR_DIR = Path(__file__).parents[2] / "data_cache" / "edgar"
YFIN_DIR = Path(__file__).parents[2] / "data_cache" / "yfin"


# --- Freshness probes -------------------------------------------------------


def _last_modified_iso(path: Path) -> str:
    if not path.exists():
        return "—"
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).strftime(
        "%Y-%m-%d %H:%M UTC"
    )


def _largest_subdir_age(parent: Path) -> tuple[int, str]:
    """Returns (count, most_recent_timestamp_str) for a parent dir of subdirs."""
    if not parent.exists():
        return 0, "—"
    subdirs = [p for p in parent.iterdir() if p.is_dir()]
    if not subdirs:
        return 0, "—"
    latest = max(subdirs, key=lambda p: p.stat().st_mtime)
    return len(subdirs), datetime.fromtimestamp(latest.stat().st_mtime, tz=UTC).strftime(
        "%Y-%m-%d %H:%M UTC"
    )


# --- Eval run loading -------------------------------------------------------


def _load_eval_runs() -> list[dict]:
    if not EVAL_DIR.exists():
        return []
    out: list[dict] = []
    for path in sorted(EVAL_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text())
            data["_filename"] = path.name
            out.append(data)
        except json.JSONDecodeError:
            continue
    return out


# --- Render helpers ---------------------------------------------------------


def _curated_universe_tickers() -> list[str]:
    """Sorted unique tickers across every curated thesis JSON in `theses/`.

    Mirrors the heartbeat sweep universe (`cio/cio.py:_curated_candidates`)
    but flattens to a unique ticker set — Mission Control wants one row per
    ticker, not per (ticker, thesis) pair. Adhoc theses are excluded; they
    aren't part of the always-on monitoring surface.
    """
    theses_dir = Path("theses")
    if not theses_dir.exists():
        return []
    tickers: set[str] = set()
    for path in theses_dir.glob("*.json"):
        if path.stem.startswith("adhoc_"):
            continue
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for t in data.get("universe") or []:
            if isinstance(t, str) and t:
                tickers.add(t.upper())
    return sorted(tickers)


def render_filings_freshness_panel() -> None:
    """Per-ticker RAG corpus freshness — latest filed dates ingested into
    the filings index. Surfaces stale ingest before a drill-in runs and the user
    wonders why the Filings agent missed last quarter's 10-Q.
    """
    from data.vectors import last_filings_by_type

    st.markdown("### Filings freshness (per ticker)")
    st.caption(
        "Latest `filed_date` per filing type across every curated thesis's "
        "universe, read from the ingest manifest in state.db. `—` means no chunks for that "
        "type are ingested yet — run `scripts/ingest_universe.py` to backfill."
    )

    tickers = _curated_universe_tickers()
    if not tickers:
        st.caption("No curated theses found in `theses/`.")
        return

    rows: list[dict] = []
    for ticker in tickers:
        by_type = last_filings_by_type(ticker)
        if not by_type:
            rows.append(
                {"ticker": ticker, "10-K": "—", "10-Q": "—", "latest": "—", "other": "—"}
            )
            continue
        # Foreign issuers file 20-F / 6-K instead of 10-K / 10-Q; collapse
        # everything outside the headline pair into one cell so the column
        # count stays stable and 20-F coverage is still visible.
        other_types = sorted(k for k in by_type if k not in {"10-K", "10-Q"})
        rows.append(
            {
                "ticker": ticker,
                "10-K": by_type.get("10-K", "—"),
                "10-Q": by_type.get("10-Q", "—"),
                "latest": max(by_type.values()),
                "other": ", ".join(f"{k}={by_type[k]}" for k in other_types) or "—",
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_freshness_panel() -> None:
    from data import state as state_db

    st.markdown("### Data-source freshness")
    cols = st.columns(4)
    with cols[0]:
        n, ts = _largest_subdir_age(EDGAR_DIR)
        freshness_card("EDGAR cached tickers", str(n), ts)
    with cols[1]:
        n_yfin = len(list(YFIN_DIR.glob("*.json"))) if YFIN_DIR.exists() else 0
        latest = max(
            (p.stat().st_mtime for p in YFIN_DIR.glob("*.json") if YFIN_DIR.exists()),
            default=None,
        )
        latest_str = (
            datetime.fromtimestamp(latest, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
            if latest
            else "—"
        )
        freshness_card("yfinance cache", str(n_yfin), latest_str)
    with cols[2]:
        manifest = state_db.ingest_manifest_summary()
        freshness_card(
            "Filings index",
            f"{manifest['tickers']} tickers · {manifest['filings']} filings · "
            f"{manifest['chunks']:,} chunks",
            manifest["last_ingested_at"] or "—",
        )
    with cols[3]:
        n_demos = len(list(DEMO_DIR.glob("*.json"))) if DEMO_DIR.exists() else 0
        latest = max(
            (p.stat().st_mtime for p in DEMO_DIR.glob("*.json") if DEMO_DIR.exists()),
            default=None,
        )
        latest_str = (
            datetime.fromtimestamp(latest, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
            if latest
            else "—"
        )
        freshness_card("Cached drill-ins", str(n_demos), latest_str)


def _normalise_score(r: dict) -> float | None:
    """Map every eval suite's score convention onto a single 0-1 numeric
    so we can chart trends across suites. Returns None when the run has
    no comparable score (e.g. a structural-counts summary row)."""
    if "score" in r and r["score"] is not None:
        # Tier 2 LLM-judge: integer 0-3 (NONE/WEAK/PARTIAL/HIGH)
        try:
            return float(r["score"]) / 3.0
        except (TypeError, ValueError):
            return None
    if "groundedness_rate" in r and r["groundedness_rate"] is not None:
        # RAG eval: already 0-1
        try:
            return float(r["groundedness_rate"])
        except (TypeError, ValueError):
            return None
    if "precision_at_k" in r and r["precision_at_k"] is not None:
        try:
            return float(r["precision_at_k"])
        except (TypeError, ValueError):
            return None
    if "ndcg_at_k" in r and r["ndcg_at_k"] is not None:
        try:
            return float(r["ndcg_at_k"])
        except (TypeError, ValueError):
            return None
    return None


def _format_score_for_display(r: dict) -> str:
    if "score" in r and r["score"] is not None:
        # Show as label + raw int for readability
        label = str(r.get("label", ""))
        return f"{label} ({r['score']}/3)" if label else str(r["score"])
    if "groundedness_rate" in r and r["groundedness_rate"] is not None:
        return f"{float(r['groundedness_rate']):.2f}"
    if "precision_at_k" in r and r["precision_at_k"] is not None:
        return f"P@K={float(r['precision_at_k']):.2f}"
    if "ndcg_at_k" in r and r["ndcg_at_k"] is not None:
        return f"NDCG={float(r['ndcg_at_k']):.2f}"
    return "—"


def _suite_trend_dataframe(suite_runs: list[dict]) -> pd.DataFrame | None:
    """Build a per-suite (timestamp → normalised-score) dataframe for the
    line chart. Returns None if no runs have a comparable score."""
    pts = []
    for r in suite_runs:
        s = _normalise_score(r)
        ts = r.get("timestamp")
        if s is None or not ts:
            continue
        pts.append({"timestamp": ts[:19], "score": s})
    if not pts:
        return None
    df = pd.DataFrame(pts).sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.set_index("timestamp")


def render_eval_runs() -> None:
    st.markdown("### Eval test runs")
    st.caption(
        "Quality-grading runs from `pytest -m eval` — automated scoring of "
        "agent outputs (Tier 2 LLM-judge, RAG retrieval, RAGAS). NOT the "
        "same as drill-in graph runs; these are unit-test artefacts. Use "
        "the per-suite expanders below to track quality trends over time."
    )
    runs = _load_eval_runs()
    if not runs:
        st.info("No eval runs recorded yet. Run `pytest -m eval` to generate.")
        return

    # Aggregate by suite
    by_suite: dict[str, list[dict]] = {}
    for r in runs:
        by_suite.setdefault(r.get("suite", "unknown"), []).append(r)

    # Top-line summary cards: one per suite, colour-coded by latest score
    st.markdown("#### Suite summary")
    cols = st.columns(min(4, max(1, len(by_suite))))
    for col, (suite, suite_runs) in zip(cols, by_suite.items(), strict=False):
        with col, st.container(border=True):
            st.markdown(f"**{suite}**")
            st.metric("Runs recorded", len(suite_runs))
            latest = suite_runs[0]  # _load_eval_runs returns reverse-sorted
            score_str = _format_score_for_display(latest)
            ts = (latest.get("timestamp") or "?")[:16].replace("T", " ")
            st.caption(f"Latest: {ts} · {score_str}")

    section_divider()

    # Per-suite expander with: trend chart + recent rows
    st.markdown("#### Per-suite trends")
    st.caption(
        "Score axis is normalised to 0-1 across suites so trends are "
        "visually comparable. Tier 2 LLM-judge maps NONE/WEAK/PARTIAL/HIGH "
        "to 0/0.33/0.66/1.0; RAG suites use their native 0-1 metric."
    )
    for suite in sorted(by_suite.keys()):
        suite_runs = by_suite[suite]
        with st.expander(f"📊 {suite} — {len(suite_runs)} run(s)"):
            trend = _suite_trend_dataframe(suite_runs)
            if trend is not None and len(trend) >= 2:
                st.line_chart(trend, height=180)
            elif trend is not None and len(trend) == 1:
                st.caption(
                    f"One run only — score: {trend['score'].iloc[0]:.2f} "
                    f"(line chart needs ≥2 points)."
                )
            else:
                st.caption("No comparable score on these runs (likely a structural-counts summary).")

            # Inline recent rows for this suite
            rows = []
            for r in suite_runs[:25]:
                rows.append(
                    {
                        "timestamp": str(r.get("timestamp", "?"))[:19],
                        "tier": str(r.get("tier", "?")),
                        "ticker": str(r.get("ticker", "?")),
                        "thesis": str(r.get("thesis", "?")),
                        "score": _format_score_for_display(r),
                        "rationale": str(r.get("rationale") or "")[:120],
                    }
                )
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    section_divider()

    # All-runs flat table (kept as before, mostly for raw scrolling)
    st.markdown("#### All recent eval runs (flat)")
    rows = []
    for r in runs[:50]:
        rows.append(
            {
                "timestamp": str(r.get("timestamp", "?"))[:19],
                "tier": str(r.get("tier", "?")),
                "suite": str(r.get("suite", "?")),
                "ticker": str(r.get("ticker", "?")),
                "thesis": str(r.get("thesis", "?")),
                "score": _format_score_for_display(r),
                "label": str(r.get("label", "—")),
                "rationale": str(r.get("rationale") or "")[:80],
            }
        )
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)


def _run_status_label(r: dict) -> str:
    """Derive the three-way status a run row renders: failed / degraded /
    completed. 'Degraded' = the graph finished but at least one agent
    failed or logged a run-scoped error (soft failure) — previously these
    looked identical to healthy runs."""
    status = str(r.get("status") or "")
    if status == "failed":
        return "❌ failed"
    if status == "completed":
        if int(r.get("failed_nodes") or 0) > 0 or int(r.get("n_errors") or 0) > 0:
            return "⚠️ degraded"
        return "✅ completed"
    return status or "?"


def _fmt_tokens(n: int) -> str:
    return f"{n / 1000:.0f}k" if n >= 1000 else str(n)


# Graph topology order — the runs table renders one status dot per agent in
# this order so a glance shows WHERE a run broke, not just that it did.
_NODE_ORDER = (
    "load_thesis", "fundamentals", "filings", "news",
    "risk", "monte_carlo", "synthesis",
)
def _within_days(iso: str, days: int) -> bool:
    try:
        ts = datetime.fromisoformat(str(iso))
    except (TypeError, ValueError):
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return (datetime.now(UTC) - ts).days < days


_MC_CSS = """
<style>
.mc-kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 4px 0 6px; }
.mc-kpi { background: #FFFFFF; border: 1px solid #E0D5C2; border-radius: 10px; padding: 14px 16px; }
.mc-kpi.alarm { border-color: #DBB3A8; background: #F6E3DD; }
.mc-kpi .lbl { font-size: 11px; letter-spacing: 0.07em; text-transform: uppercase; color: #6B6152; }
.mc-kpi .val { font: 600 25px/1.1 var(--mc-mono, ui-monospace, "SF Mono", Menlo, monospace); color: #1A1611; margin-top: 3px; font-variant-numeric: tabular-nums; }
.mc-kpi.alarm .val { color: #A33D2E; }
.mc-kpi .hint { font-size: 12px; color: #6B6152; margin-top: 2px; }
.mc-spark { display: flex; align-items: flex-end; gap: 3px; height: 26px; margin-top: 7px; }
.mc-spark i { flex: 1; background: #E0D5C2; border-radius: 2px 2px 0 0; min-height: 2px; }
.mc-spark i.today { background: #2D4F3A; }

/* Runs table — bespoke HTML rows so the dots, pills and stripe match the
   mockup; the "Open" control per row is a real Streamlit button beside it. */
.mcrow { display: grid;
  grid-template-columns: 104px minmax(90px, 1.2fr) 66px 112px 100px 58px 74px;
  align-items: center; gap: 8px; border-left: 3px solid transparent; padding: 0 4px 0 9px;
  border-bottom: 1px solid #EDE5D5; min-height: 44px; font-size: 13px; overflow: hidden; }
.mcrow > span { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.mcrow.fail { border-left-color: #A33D2E; background: #F6E3DD; }
.mcrow.deg  { border-left-color: #9A6B1F; }
.mchead { border-bottom: 1px solid #E0D5C2; min-height: 34px; border-left-color: transparent; }
.mchead span { font: 600 10.5px sans-serif; letter-spacing: 0.07em; text-transform: uppercase; color: #6B6152; }
.mcrow .mono { font-family: var(--mc-mono, ui-monospace, "SF Mono", Menlo, monospace); font-variant-numeric: tabular-nums; color: #1A1611; }
.mcrow .tk { font-weight: 600; color: #1A1611; }
.mcrow .muted { color: #6B6152; font-size: 12px; }
.mcrow .dot { display: inline-block; width: 11px; height: 11px; border-radius: 50%; margin-right: 3px; }
.mcrow .dot.ok { background: #2D4F3A; }
.mcrow .dot.fail { background: #A33D2E; }
.mcrow .dot.none { background: #FFFFFF; border: 1.5px solid #E0D5C2; }
.mcrow .tpill { font: 500 10.5px var(--mc-mono, ui-monospace, "SF Mono", Menlo, monospace);
  color: #6B6152; border: 1px solid #E0D5C2; border-radius: 4px; padding: 1px 6px; }
.mcrow .spill { font: 600 10px sans-serif; letter-spacing: 0.04em; text-transform: uppercase;
  border-radius: 4px; padding: 2px 8px; }
.mcrow .spill.ok { background: #E4EAE2; color: #2D4F3A; }
.mcrow .spill.deg { background: #F5EBD2; color: #9A6B1F; border: 1px solid #DCC791; }
.mcrow .spill.fail { background: #A33D2E; color: #FFFFFF; }
</style>
"""

_RUN_COLS = [11, 1.6]  # [styled row cell, Open button] — keeps buttons aligned.


def _dots_html(status_map: dict[str, str]) -> str:
    """Colored status dots in graph order (sage ok / brick failed / hollow
    didn't-run) — the mockup's AGENTS strip."""
    cls = {"completed": "ok", "failed": "fail"}
    return "".join(
        f'<span class="dot {cls.get(status_map.get(n, ""), "none")}"></span>'
        for n in _NODE_ORDER
    )


def _status_bits(r: dict) -> tuple[str, str, str]:
    """(pill text, pill css class, row css class) for a run's three-way
    status. Mirrors _run_status_label without the emoji."""
    status = str(r.get("status") or "")
    if status == "failed":
        return "failed", "fail", "fail"
    if status == "completed":
        if int(r.get("failed_nodes") or 0) > 0 or int(r.get("n_errors") or 0) > 0:
            return "degraded", "deg", "deg"
        return "completed", "ok", ""
    return (status or "?"), "ok", ""


def _render_runs_table(runs_all: list[dict]) -> None:
    """Filter chips + per-agent status dots + row-select → Run Inspector.
    Standalone so an early return here never skips the CIO panels that
    render after it on the page."""
    from data import state as state_db

    st.markdown("#### Recent drill-in runs")
    if not runs_all:
        st.caption("No runs recorded.")
        return

    # Classify manual vs CIO-triggered from the CIO action log (one query):
    # any run_id a drill/reuse decision points at was CIO-ordered.
    cio_run_ids = set()
    for a in state_db.recent_cio_actions(limit=500):
        for key in ("drill_run_id", "reuse_run_id"):
            if a.get(key):
                cio_run_ids.add(a[key])

    choice = st.pills(
        "Filter",
        ["All", "Failed", "Degraded", "Completed", "Manual", "CIO-triggered"],
        default="All",
        label_visibility="collapsed",
        key="mc_run_filter",
    ) or "All"

    def _keep(r: dict) -> bool:
        label = _run_status_label(r)
        rid = r.get("run_id")
        if choice == "Failed":
            return "failed" in label
        if choice == "Degraded":
            return "degraded" in label
        if choice == "Completed":
            return "completed" in label
        if choice == "Manual":
            return rid not in cio_run_ids
        if choice == "CIO-triggered":
            return rid in cio_run_ids
        return True

    filtered = [r for r in runs_all if _keep(r)][:25]
    if not filtered:
        st.caption(f"No runs match “{choice}”.")
        return

    statuses = state_db.node_status_by_run([r.get("run_id") for r in filtered])
    st.caption(
        "AGENTS strip, left→right: load · fundamentals · filings · news · "
        "risk · monte_carlo · synthesis  (● ok · ● failed · ○ didn't run). "
        "Hit **Open** to drill into a run."
    )

    # Header row (aligned to the same column split as each data row).
    head = st.columns(_RUN_COLS, vertical_alignment="center")
    head[0].markdown(
        '<div class="mcrow mchead"><span>Started</span><span>Run</span>'
        "<span>Trigger</span><span>Agents</span><span>Status</span>"
        "<span>Duration</span><span>Cost</span></div>",
        unsafe_allow_html=True,
    )
    head[1].markdown("&nbsp;", unsafe_allow_html=True)

    for r in filtered:
        rid = str(r.get("run_id") or "")
        text, spill, rowcls = _status_bits(r)
        # MM-DD HH:MM (drop the year) — matches the mockup and saves width.
        started = str(r.get("started_at") or "")[5:16].replace("T", " ")
        thesis = str(r.get("thesis") or "?")
        dur = f"{r['duration_s']:.0f}s" if r.get("duration_s") else "—"
        cost = f"${float(r.get('cost_usd') or 0.0):.4f}"
        trig = "🤖 cio" if rid in cio_run_ids else "🖱 manual"
        row_html = (
            f'<div class="mcrow {rowcls}">'
            f'<span class="mono">{started}</span>'
            f'<span><span class="tk">{r.get("ticker") or "?"}</span>'
            f'<span class="muted"> · {thesis}</span></span>'
            f'<span><span class="tpill">{trig}</span></span>'
            f'<span>{_dots_html(statuses.get(rid, {}))}</span>'
            f'<span><span class="spill {spill}">{text}</span></span>'
            f'<span class="mono">{dur}</span>'
            f'<span class="mono">{cost}</span>'
            f"</div>"
        )
        cols = st.columns(_RUN_COLS, vertical_alignment="center")
        cols[0].markdown(row_html, unsafe_allow_html=True)
        if cols[1].button("Open →", key=f"open_{rid}", use_container_width=True):
            st.session_state["inspect_run_id"] = rid
            st.switch_page("pages/run_inspector.py")


def render_state_db_panel() -> None:
    """Step 5z observability — reads from data/state.py SQLite telemetry."""
    st.markdown("### Drill-in runs")
    st.caption(
        "Every full LangGraph drill-in (the dashboard's 🔍 Run drill-in "
        "button), with timing, status, cost, and per-node telemetry. Backed "
        "by `data_cache/state.db`. **Select a row to open it in the Run "
        "Inspector** (agent timeline, errors, LLM call trace)."
    )
    from data import state as state_db

    # Read the runtime DB_PATH (set in conftest fixtures during tests, or
    # the project default in production). Hard-coding the path here would
    # bypass the test fixture and break Mission Control's smoke test.
    if not Path(state_db.DB_PATH).exists():
        st.info(
            "No graph runs recorded yet. Run a drill-in from the dashboard — "
            "it'll write a `graph_runs` row + per-node telemetry to "
            "`data_cache/state.db`."
        )
        return

    st.markdown(_MC_CSS, unsafe_allow_html=True)

    summary = state_db.health_summary()
    spend = state_db.cost_today()
    # One fetch drives the KPIs, the filter counts, and the table below.
    runs_all = state_db.recent_runs(limit=100)
    last7 = [r for r in runs_all if _within_days(r.get("started_at") or "", 7)]
    labels7 = [_run_status_label(r) for r in last7]
    n_completed = sum(1 for x in labels7 if "completed" in x)
    n_degraded = sum(1 for x in labels7 if "degraded" in x)
    n_failed = sum(1 for x in labels7 if "failed" in x)

    # Spend sparkline — last 7 days of node_runs cost, today's bar in sage.
    cost7 = state_db.daily_cost(days=7)
    costs = [float(c.get("cost_usd") or 0.0) for c in cost7]
    cmax = max(costs, default=0.0) or 1.0
    today_iso = datetime.now(UTC).date().isoformat()
    spark = "".join(
        f'<i class="{"today" if (c.get("date") == today_iso) else ""}" '
        f'style="height:{max(2, round(float(c.get("cost_usd") or 0.0) / cmax * 24))}px"></i>'
        for c in cost7
    ) or '<i style="height:2px"></i>'

    last_run = (summary["last_run_at"] or "—")[:16].replace("T", " ")
    last_is_fail = bool(last7) and "failed" in labels7[0]
    rate = summary["failure_rate_7d"]
    rate_str = f"{rate:.0%}" if rate is not None else "—"

    st.markdown(
        f"""
        <div class="mc-kpis">
          <div class="mc-kpi">
            <div class="lbl">Spend today</div>
            <div class="val">${spend['cost_usd']:.2f}</div>
            <div class="mc-spark">{spark}</div>
            <div class="hint">last 7 days · node_runs.cost_usd</div>
          </div>
          <div class="mc-kpi">
            <div class="lbl">Runs · 7d</div>
            <div class="val">{len(last7)}</div>
            <div class="hint">🟢 {n_completed} completed · ⚠️ {n_degraded} degraded · 🔴 {n_failed} failed</div>
          </div>
          <div class="mc-kpi{' alarm' if last_is_fail else ''}">
            <div class="lbl">Last run</div>
            <div class="val">{'FAILED' if last_is_fail else 'OK'}</div>
            <div class="hint">{last_run} UTC · {summary['total_runs']} total</div>
          </div>
          <div class="mc-kpi">
            <div class="lbl">Failure rate · 7d</div>
            <div class="val">{rate_str}</div>
            <div class="hint">graph_runs marked failed / total</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    proj = os.environ.get("LANGSMITH_PROJECT", "")
    if proj and os.environ.get("LANGSMITH_TRACING", "").lower() == "true":
        st.link_button(
            "🔗 Open LangSmith",
            f"https://smith.langchain.com/o/-/projects/p/{proj}",
        )

    section_divider()

    # Daily-runs + daily-cost charts, side by side.
    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.markdown("#### Daily run counts (last 14 days)")
        daily = state_db.daily_run_counts(days=14)
        if daily:
            df = pd.DataFrame(daily).set_index("day")
            st.bar_chart(df[["completed", "failed"]])
        else:
            st.caption("No daily-run data yet.")
    with chart_cols[1]:
        st.markdown("#### Daily LLM cost (last 14 days)")
        cost_rows = state_db.daily_cost(days=14)
        if cost_rows:
            cost_df = pd.DataFrame(cost_rows).set_index("date")
            st.bar_chart(cost_df[["cost_usd"]])
        else:
            st.caption("No cost data yet.")

    section_divider()

    # Recent runs table — filter chips + agent status dots + row-select.
    # In its own helper so an empty-runs / empty-filter early return skips
    # only the table, not the CIO panels rendered afterwards.
    _render_runs_table(runs_all)

    section_divider()

    # Recent CIO actions (Step 11.14 + 11.19 telemetry columns)
    st.markdown("#### Recent CIO actions")
    st.caption(
        "Each row is one drill / reuse / dismiss decision the CIO made on a "
        "heartbeat or `/cio` invocation. Model + cost + latency capture the "
        "LLM call that produced the decision. `source` says who decided: "
        "`llm`, `budget_cap` (demoted drill) or `fallback` (planner error). "
        "Yo-yo-guard shortcuts (`gate`) are hidden — they'd otherwise fill "
        "the table with one cycle's 'nothing changed' rows."
    )
    # Pull a few cycles' worth so the 25 shown are real judgements, not
    # the tail of the latest sweep's gate shortcuts.
    cio_actions = [
        a for a in state_db.recent_cio_actions(limit=400)
        if a.get("source") != "gate"
    ][:25]
    if cio_actions:
        action_rows = []
        for a in cio_actions:
            decision_run = a.get("drill_run_id") or a.get("reuse_run_id") or "—"
            cost = float(a.get("cost_usd") or 0.0)
            lat = float(a.get("latency_s") or 0.0)
            action_rows.append(
                {
                    "ts": str(a.get("ts") or "")[:19].replace("T", " "),
                    "trigger": str(a.get("trigger") or ""),
                    "ticker": str(a.get("ticker") or ""),
                    "thesis": str(a.get("thesis") or "—"),
                    "action": str(a.get("action") or ""),
                    "source": str(a.get("source") or "—"),
                    "confidence": str(a.get("confidence") or "—"),
                    "model": str(a.get("model_used") or "—"),
                    "tok_in": int(a.get("tokens_in") or 0),
                    "tok_out": int(a.get("tokens_out") or 0),
                    "cost_$": f"${cost:.4f}" if cost > 0 else "—",
                    "lat_s": f"{lat:.2f}" if lat > 0 else "—",
                    "rationale": str(a.get("rationale") or "")[:100],
                    "run_id": str(decision_run)[:8],
                }
            )
        st.dataframe(pd.DataFrame(action_rows), use_container_width=True, hide_index=True)
    else:
        st.caption(
            "No CIO actions recorded. Run `/cio` from Telegram or wait "
            "for the heartbeat (5am + 1pm PT)."
        )

    # Recent CIO cycles (the meta-rollup row, now with model + total cost).
    st.markdown("#### Recent CIO cycles")
    cio_runs = state_db.recent_cio_runs(limit=10)
    if cio_runs:
        cycle_rows = []
        for r in cio_runs:
            total_cost = float(r.get("total_cost_usd") or 0.0)
            cycle_rows.append(
                {
                    "started": str(r.get("started_at") or "")[:19].replace("T", " "),
                    "trigger": str(r.get("trigger") or ""),
                    "status": str(r.get("status") or ""),
                    "model": str(r.get("model_used") or "—"),
                    "actions": int(r.get("n_actions") or 0),
                    "drilled": int(r.get("n_drilled") or 0),
                    "reused": int(r.get("n_reused") or 0),
                    "dismissed": int(r.get("n_dismissed") or 0),
                    "cost_$": f"${total_cost:.4f}" if total_cost > 0 else "—",
                    "duration_s": (
                        f"{r['duration_s']:.1f}" if r.get("duration_s") else "—"
                    ),
                }
            )
        st.dataframe(pd.DataFrame(cycle_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No CIO cycles recorded yet.")

    # Step 11.19 — Per-model performance rollup (the comparison panel
    # the user wants for "which model gave me the best decisions for the
    # least money over the last 30 days").
    st.markdown("#### CIO model performance — last 30 days")
    st.caption(
        "Compare models across the same per-(ticker, thesis) decision surface. "
        "`parse_fail` counts deterministic-fallback dismisses where the model's "
        "JSON output was unparseable. Lower `parse_fail` and lower `cost / call` "
        "is better, all else equal."
    )
    perf = state_db.cio_model_performance(days=30)
    if perf:
        perf_rows = []
        for p in perf:
            n_calls = int(p.get("n_calls") or 0)
            tot = float(p.get("total_cost_usd") or 0.0)
            avg_lat = float(p.get("avg_latency_s") or 0.0)
            n_fail = int(p.get("n_parse_fails") or 0)
            fail_rate = (n_fail / n_calls * 100.0) if n_calls else 0.0
            cost_per_call = (tot / n_calls) if n_calls else 0.0
            perf_rows.append(
                {
                    "model": p.get("model_used") or "—",
                    "calls": n_calls,
                    "drills": int(p.get("n_drills") or 0),
                    "reuses": int(p.get("n_reuses") or 0),
                    "dismisses": int(p.get("n_dismisses") or 0),
                    "parse_fail %": f"{fail_rate:.1f}",
                    "avg lat s": f"{avg_lat:.2f}",
                    "tot $": f"${tot:.4f}",
                    "$ / call": f"${cost_per_call:.5f}",
                    "avg tok_in": int(p.get("avg_tokens_in") or 0),
                    "avg tok_out": int(p.get("avg_tokens_out") or 0),
                }
            )
        st.dataframe(pd.DataFrame(perf_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No CIO LLM calls recorded in the last 30 days.")

    section_divider()

    # Recent errors table
    st.markdown("#### Recent errors")
    errs = state_db.recent_errors(limit=20)
    if errs:
        rows = []
        for e in errs:
            rows.append(
                {
                    "ts": str(e.get("ts") or "")[:19].replace("T", " "),
                    "agent": str(e.get("agent") or ""),
                    "message": str(e.get("message") or "")[:120],
                    "run_id": str(e.get("run_id") or "—")[:8],
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No errors logged. (Good news.)")


def render_recent_demos() -> None:
    st.markdown("### Cached drill-in runs")
    if not DEMO_DIR.exists() or not list(DEMO_DIR.glob("*.json")):
        st.caption("No cached drill-ins yet. Run one from the dashboard.")
        return
    rows = []
    for p in sorted(DEMO_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        rows.append(
            {
                "filename": str(p.name),
                "ticker": str(data.get("ticker", "?")),
                "thesis": str((data.get("thesis") or {}).get("name", "?")),
                "confidence": str(data.get("synthesis_confidence", "?")),
                "saved": datetime.fromtimestamp(
                    p.stat().st_mtime, tz=UTC
                ).strftime("%Y-%m-%d %H:%M UTC"),
                "errors": int(len(data.get("errors") or [])),
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_notion_panel() -> None:
    """Show whether Notion is configured + the most recent reports written.
    Notion is the long-term content memory (Step 9). When unconfigured,
    explain how to set it up; when configured, deep-link to recent reports."""
    st.markdown("### Notion (long-term memory)")
    try:
        from data import notion as _notion
    except ImportError:
        st.warning("`notion-client` not installed. Add it to requirements.txt.")
        return

    if not _notion.is_configured():
        st.info(
            "Notion is **not configured**. To enable long-term memory of every "
            "drill-in (reports, watchlist, alerts), follow the setup steps in "
            "`scripts/bootstrap_notion.py`'s docstring + "
            "`docs/ARCHITECTURE.md` §12.\n\n"
            "When `NOTION_API_KEY` is set in `.env`, every drill-in writes its "
            "report to your Notion Reports database in the background — "
            "this panel will then surface a deep-link list."
        )
        return

    st.caption(
        "Notion is connected. Every drill-in writes the synthesis report + "
        "watchlist items to your workspace as a sidecar of the runner thread. "
        "Failures are logged but never block the dashboard."
    )

    # Last 10 reports — best-effort; surface errors as a caption if the
    # query fails (Notion outage shouldn't break Mission Control).
    try:
        reports = _notion.read_recent_reports(limit=10)
    except Exception as e:
        st.warning(f"Could not query Notion Reports DB: {e}")
        return

    if not reports:
        st.caption(
            "Reports DB is empty. Run a drill-in from the dashboard — the "
            "report will appear here within a few seconds of completion."
        )
        return

    rows = []
    for r in reports:
        rows.append(
            {
                "title": r.get("title") or "—",
                "ticker": r.get("ticker") or "?",
                "thesis": r.get("thesis") or "?",
                "confidence": r.get("confidence") or "—",
                "date": r.get("date") or "—",
                "url": r.get("url") or "",
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "url": st.column_config.LinkColumn("Open in Notion", display_text="Open ↗")
        },
    )


def main() -> None:
    page_header(
        "Mission Control",
        subtitle=(
            "System health at a glance. Eval pass/fail, data-source freshness, "
            "graph-run history. Reads `data_cache/eval/runs/` + (Step 5z) "
            "`data_cache/state.db`."
        ),
    )
    render_freshness_panel()
    section_divider()
    render_filings_freshness_panel()
    section_divider()
    render_state_db_panel()
    section_divider()
    render_notion_panel()
    section_divider()
    render_eval_runs()
    section_divider()
    render_recent_demos()


main()
