"""Mission Control page — at-a-glance system health.

Reads from existing artefacts (no Step 5z dependency):
  - data_cache/eval/runs/*.json — eval run history (Tier 1 + Tier 2 + RAGAS)
  - data_cache/edgar/ — last-touched timestamps per ticker
  - data_cache/yfin/ — last-touched per ticker
  - data_cache/chroma/ — collection size + last modified
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
CHROMA_DIR = Path(__file__).parents[2] / "data_cache" / "chroma"


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


def render_freshness_panel() -> None:
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
        if CHROMA_DIR.exists():
            size_bytes = sum(f.stat().st_size for f in CHROMA_DIR.rglob("*") if f.is_file())
            freshness_card(
                "ChromaDB",
                f"{size_bytes / 1e6:.1f} MB",
                _last_modified_iso(CHROMA_DIR),
            )
        else:
            freshness_card("ChromaDB", "—", "—")
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


def render_state_db_panel() -> None:
    """Step 5z observability — reads from data/state.py SQLite telemetry."""
    st.markdown("### Drill-in runs")
    st.caption(
        "Every full LangGraph drill-in (the dashboard's 🔍 Run drill-in "
        "button), with timing, status, and per-node telemetry. Backed by "
        "`data_cache/state.db`."
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

    summary = state_db.health_summary()
    cols = st.columns(4)
    with cols[0]:
        cols[0].metric("Total graph runs", summary["total_runs"])
    with cols[1]:
        cols[1].metric(
            "Last run",
            (summary["last_run_at"] or "—")[:19].replace("T", " "),
        )
    with cols[2]:
        rate = summary["failure_rate_7d"]
        cols[2].metric(
            "Failure rate (7d)",
            f"{rate:.0%}" if rate is not None else "—",
        )
    with cols[3]:
        # Quick LangSmith deep-link if the user has it configured.
        proj = os.environ.get("LANGSMITH_PROJECT", "")
        if proj and os.environ.get("LANGSMITH_TRACING", "").lower() == "true":
            cols[3].link_button(
                "🔗 LangSmith",
                f"https://smith.langchain.com/o/-/projects/p/{proj}",
                use_container_width=True,
            )
        else:
            cols[3].caption("LangSmith disabled")

    section_divider()

    # Daily-runs chart
    st.markdown("#### Daily run counts (last 14 days)")
    daily = state_db.daily_run_counts(days=14)
    if daily:
        df = pd.DataFrame(daily).set_index("day")
        st.bar_chart(df[["completed", "failed"]])
    else:
        st.caption("No daily-run data yet.")

    section_divider()

    # Recent runs table
    st.markdown("#### Recent drill-in runs")
    runs = state_db.recent_runs(limit=25)
    if runs:
        rows = []
        for r in runs:
            rows.append(
                {
                    "started": str(r.get("started_at") or "")[:19].replace("T", " "),
                    "ticker": str(r.get("ticker") or "?"),
                    "thesis": str(r.get("thesis") or "?"),
                    "status": str(r.get("status") or ""),
                    "confidence": str(r.get("confidence") or "—"),
                    "duration_s": (
                        f"{r['duration_s']:.1f}" if r.get("duration_s") else "—"
                    ),
                    "nodes": int(r.get("node_runs_count") or 0),
                    "failed": int(r.get("failed_nodes") or 0),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No runs recorded.")

    section_divider()

    # Recent CIO actions (Step 11.14 + 11.19 telemetry columns)
    st.markdown("#### Recent CIO actions")
    st.caption(
        "Each row is one drill / reuse / dismiss decision the CIO made on a "
        "heartbeat or `/cio` invocation. Model + cost + latency capture the "
        "LLM call that produced the decision."
    )
    cio_actions = state_db.recent_cio_actions(limit=25)
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
    render_state_db_panel()
    section_divider()
    render_notion_panel()
    section_divider()
    render_eval_runs()
    section_divider()
    render_recent_demos()


main()
