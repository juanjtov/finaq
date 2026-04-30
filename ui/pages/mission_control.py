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


def render_eval_runs() -> None:
    st.markdown("### Eval runs")
    runs = _load_eval_runs()
    if not runs:
        st.info("No eval runs recorded yet. Run `pytest -m eval` to generate.")
        return

    # Aggregate counts by suite
    by_suite: dict[str, list[dict]] = {}
    for r in runs:
        by_suite.setdefault(r.get("suite", "unknown"), []).append(r)

    cols = st.columns(min(4, len(by_suite)))
    for col, (suite, suite_runs) in zip(cols, by_suite.items(), strict=False):
        with col, st.container(border=True):
            st.markdown(f"**{suite}**")
            st.metric("Runs recorded", len(suite_runs))
            latest = suite_runs[0]
            st.caption(f"Latest: {latest.get('timestamp', '?')[:19]}")

    section_divider()
    st.markdown("#### Recent runs")
    rows = []
    for r in runs[:25]:
        # Eval suites use different score conventions: Tier 2 LLM-judge uses
        # an integer `score` (0-3) plus `label`; RAG eval uses a float
        # `groundedness_rate` (0..1). Coerce all to a single string column so
        # pyarrow / Streamlit's dataframe renderer doesn't choke on mixed
        # types.
        if "score" in r and r["score"] is not None:
            score_str = str(r["score"])
        elif "groundedness_rate" in r and r["groundedness_rate"] is not None:
            score_str = f"{float(r['groundedness_rate']):.2f}"
        else:
            score_str = "—"
        rows.append(
            {
                "timestamp": str(r.get("timestamp", "?"))[:19],
                "tier": str(r.get("tier", "?")),
                "suite": str(r.get("suite", "?")),
                "ticker": str(r.get("ticker", "?")),
                "thesis": str(r.get("thesis", "?")),
                "score": score_str,
                "label": str(r.get("label", "—")),
                "rationale": str(r.get("rationale") or "")[:80],
            }
        )
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_state_db_panel() -> None:
    """Step 5z observability — reads from data/state.py SQLite telemetry."""
    st.markdown("### Graph runs (Step 5z observability)")
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
    st.markdown("#### Recent runs")
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
    render_eval_runs()
    section_divider()
    render_state_db_panel()
    section_divider()
    render_recent_demos()


main()
