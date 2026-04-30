"""Architecture page — what FINAQ is and how its parts fit together.

Auto-discovered by Streamlit from `ui/pages/`. Reads from existing source
files (no hand-curated content store) so the page never rots:
  - LangGraph topology → mermaid via `build_graph().get_graph().draw_mermaid()`
  - Agent cards from `agents/*.py` + `utils/models.py` model strings
  - Data sources hard-coded (one place to maintain)
  - Test tiers + current model strings shown live
"""

from __future__ import annotations

# Bootstrap (see ui/app.py for explanation).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os

import streamlit as st
import streamlit.components.v1 as components

from ui.components import freshness_card, page_header, section_divider

st.set_page_config(page_title="FINAQ — Architecture", page_icon="🏗️", layout="wide")


# --- Static data describing the platform -----------------------------------

AGENT_CARDS = [
    {
        "name": "Fundamentals",
        "model_env": "MODEL_FUNDAMENTALS",
        "role": "Computes 5-year revenue CAGR, gross/operating margins, FCF trajectory, sector-relative valuation. Produces `projections` for the Monte Carlo engine.",
        "inputs": "ticker · thesis · yfinance financials",
        "outputs": "summary · kpis · projections (revenue/margin/multiple) · evidence",
        "external_calls": "yfinance (cached 24h)",
    },
    {
        "name": "Filings",
        "model_env": "MODEL_FILINGS",
        "role": "Hybrid RAG (semantic + BM25 + RRF) over the company's recent 10-K + 10-Qs. Three subqueries per drill-in: risk factors, MD&A trajectory, segment performance.",
        "inputs": "ticker · thesis · ChromaDB filings collection",
        "outputs": "summary · risk_themes · mdna_quotes · evidence",
        "external_calls": "ChromaDB · OpenRouter embeddings",
    },
    {
        "name": "News",
        "model_env": "MODEL_NEWS",
        "role": "90-day Tavily search → 5-7 catalysts (bull) + 5-7 concerns (bear) + sentiment + URL + published date.",
        "inputs": "ticker · company_name (yfinance) · thesis",
        "outputs": "summary · catalysts · concerns · evidence",
        "external_calls": "Tavily search API",
    },
    {
        "name": "Risk",
        "model_env": "MODEL_RISK",
        "role": "Synthesis-only. Reads fundamentals/filings/news outputs, finds convergent signals, threshold breaches, divergent signals, and implicit gaps. Emits categorical level (LOW/MODERATE/ELEVATED/HIGH/CRITICAL) and derived 0–10 score.",
        "inputs": "fundamentals · filings · news (no external calls)",
        "outputs": "level · score_0_to_10 · top_risks · convergent_signals · threshold_breaches",
        "external_calls": "None",
    },
    {
        "name": "Monte Carlo",
        "model_env": "(no LLM)",
        "role": "Vectorized NumPy hybrid Owner-Earnings DCF + Multiple-based fair-value distribution. 10,000 sims × 10y horizon. Discount rate = clip(10y Treasury + per-thesis ERP, [floor, cap]).",
        "inputs": "fundamentals.projections · thesis.valuation · 10y Treasury yield",
        "outputs": "method · dcf{p10..p90} · multiple{p10..p90} · convergence_ratio · discount_rate_used",
        "external_calls": "yfinance ^TNX (cached 24h)",
    },
    {
        "name": "Synthesis",
        "model_env": "MODEL_SYNTHESIS",
        "role": "Final report writer. Reads the entire FinaqState. Produces a 9-section markdown report (CLAUDE.md §11) plus structured `confidence`, `gaps`, and `watchlist` fields.",
        "inputs": "Full state (all upstream agents)",
        "outputs": "report (markdown) · confidence · gaps · watchlist",
        "external_calls": "None (state-only)",
    },
    {
        "name": "Per-agent Q&A (`ask`)",
        "model_env": "MODEL_AGENT_QA",
        "role": "Free-text Q&A on any of the 4 worker agents. Reused by the dashboard's Direct Agent panel and the Phase 1 Telegram /fundamentals|/filings|/news|/risk commands.",
        "inputs": "state · agent name · question",
        "outputs": "AgentAnswer (answer · citations)",
        "external_calls": "OpenRouter (Filings also re-runs ChromaDB RAG)",
    },
]

DATA_SOURCES = [
    {
        "name": "SEC EDGAR",
        "module": "data/edgar.py",
        "role": "Idempotent download of recent 10-K + 10-Q filings to data_cache/edgar/",
        "freshness": "On-demand; once downloaded, cached forever",
    },
    {
        "name": "yfinance",
        "module": "data/yfin.py",
        "role": "Financial statements + price history. 24h JSON cache.",
        "freshness": "24h TTL; bypass with cache_format_version bump",
    },
    {
        "name": "ChromaDB",
        "module": "data/chroma.py",
        "role": "Persistent vector store for filings chunks. Single 'filings' collection with ticker metadata. Cosine distance.",
        "freshness": "Updated when scripts/ingest_universe.py runs",
    },
    {
        "name": "Tavily",
        "module": "data/tavily.py",
        "role": "News search API for catalyst extraction. 90-day window.",
        "freshness": "Real-time on each /news call",
    },
    {
        "name": "Treasury (10y)",
        "module": "data/treasury.py",
        "role": "Discount-rate input for Monte Carlo via yfinance ^TNX. Cached 24h.",
        "freshness": "24h TTL; falls back to 4.5% on failure",
    },
    {
        "name": "Sector multiples",
        "module": "data/sector_multiples.json",
        "role": "Hardcoded P/E centroids by sector (Damodaran NYU Stern, manual quarterly refresh).",
        "freshness": "Manual quarterly update",
    },
]

TEST_TIERS = [
    {
        "tier": "Tier 1 — deterministic",
        "scope": "Always-on. No external calls. Schema validation, math correctness, prompt assembly, structural checks.",
        "command": "pytest -m 'not integration and not eval'",
        "cost": "Free",
    },
    {
        "tier": "Tier 2 — LLM-judge",
        "scope": "Categorical (NONE/WEAK/PARTIAL/HIGH) judges for faithfulness, thesis-awareness, action specificity, etc. Rationale-first JSON.",
        "command": "pytest -m eval",
        "cost": "~$0.025 per agent suite",
    },
    {
        "tier": "Tier 3 — integration",
        "scope": "Real graph runs against real APIs (yfinance, EDGAR, Tavily, OpenRouter, Treasury). Catches plumbing bugs.",
        "command": "pytest -m integration",
        "cost": "~$0.50 per full graph run",
    },
    {
        "tier": "RAGAS",
        "scope": "Faithfulness / context_precision / context_recall / answer_relevancy on Filings + News.",
        "command": "pytest -m eval tests/test_rag_eval_ragas.py tests/test_news_eval_ragas.py",
        "cost": "~$1+ per suite",
    },
]


# --- Page render ------------------------------------------------------------


def _model_string(env_var: str) -> str:
    """Resolve a MODEL_* env var to its current value, with safe fallback."""
    if env_var.startswith("(") or env_var == "":
        return env_var
    return os.environ.get(env_var) or "(not set)"


def render_topology() -> None:
    """Render the LangGraph topology as an actual visual diagram.

    Streamlit doesn't natively render mermaid in markdown — pasting a
    mermaid fenced block shows it as a code snippet. We embed mermaid.js
    via `st.components.v1.html` with the script loaded from a CDN. No new
    Python dependency; the diagram is drawn client-side.
    """
    st.markdown("### LangGraph topology")
    st.caption(
        "Each node is wrapped in `_safe_node` — exceptions are routed into "
        "`state.errors` so a single failed agent never crashes the graph. "
        "Step 5z added per-node telemetry to `data_cache/state.db`."
    )
    try:
        from agents import build_graph

        mermaid = build_graph().get_graph().draw_mermaid()
    except Exception as e:
        st.warning(f"Could not render topology: {e}")
        return

    # Mermaid supports a `themeVariables` dict — feed our palette in so the
    # diagram visually matches the rest of the dashboard.
    html = f"""
    <div style="background:#FBF5E8; padding:1rem; border-radius:6px;
        border:1px solid #E0D5C2;">
      <pre class="mermaid">
{mermaid}
      </pre>
    </div>
    <script type="module">
      import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
      mermaid.initialize({{
        startOnLoad: true,
        theme: "base",
        themeVariables: {{
          primaryColor: "#F4ECDC",
          primaryTextColor: "#1A1611",
          primaryBorderColor: "#2D4F3A",
          lineColor: "#2D4F3A",
          secondaryColor: "#FBF5E8",
          tertiaryColor: "#FFFFFF",
          fontSize: "14px",
        }},
      }});
    </script>
    """
    components.html(html, height=520, scrolling=True)

    with st.expander("Show mermaid source"):
        st.code(mermaid, language="text")


def render_agent_cards() -> None:
    st.markdown("### Agents")
    st.caption(
        "Each agent owns one cognitive step. Model strings live in `.env`; "
        "swap one and the entire pipeline picks up the new model on restart."
    )
    for card in AGENT_CARDS:
        with st.container(border=True):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"#### {card['name']}")
                st.markdown(card["role"])
                st.markdown(
                    f"- **Inputs:** {card['inputs']}\n"
                    f"- **Outputs:** {card['outputs']}\n"
                    f"- **External calls:** {card['external_calls']}"
                )
            with col_b:
                st.caption("Model")
                st.code(_model_string(card["model_env"]), language=None)
                if card["model_env"] != "(no LLM)":
                    st.caption(f"_env var:_ `{card['model_env']}`")


def render_data_sources() -> None:
    st.markdown("### Data sources")
    cols = st.columns(2)
    for i, ds in enumerate(DATA_SOURCES):
        with cols[i % 2], st.container(border=True):
            st.markdown(f"#### {ds['name']}")
            st.caption(ds["module"])
            st.markdown(ds["role"])
            st.caption(f"_Freshness:_ {ds['freshness']}")


def render_test_tiers() -> None:
    st.markdown("### Test tiers")
    st.caption(
        "Each tier is a hard gate per CLAUDE.md §16.5. Failures are root-caused "
        "and fixed; tests are never `xfail`-ed to advance."
    )
    for t in TEST_TIERS:
        with st.container(border=True):
            st.markdown(f"**{t['tier']}**")
            st.markdown(t["scope"])
            st.code(t["command"], language="bash")
            st.caption(f"_Cost:_ {t['cost']}")


def render_palette() -> None:
    st.markdown("### Palette (CLAUDE.md §13)")
    palette = [
        ("Brand accent — Botanical sage", "#2D4F3A"),
        ("Hero / closing — Parchment", "#F4ECDC"),
        ("Card / body — White", "#FFFFFF"),
        ("Inner panel — Eggshell", "#FBF5E8"),
        ("Outer border — Taupe", "#E0D5C2"),
        ("Internal divider — Bone", "#EDE5D5"),
        ("All text — Ink", "#1A1611"),
    ]
    cols = st.columns(len(palette))
    for col, (label, hex_) in zip(cols, palette, strict=False):
        with col:
            st.markdown(
                f"""
                <div style="background:{hex_}; height:60px; border-radius:6px;
                    border:1px solid #E0D5C2;"></div>
                <div style="font-size:0.7rem; margin-top:0.4rem; line-height:1.2;
                    color:#1A1611;">{label}<br><code>{hex_}</code></div>
                """,
                unsafe_allow_html=True,
            )


def render_phase_scope() -> None:
    st.markdown("### Phase scope")
    cols = st.columns(3)
    with cols[0], st.container(border=True):
        st.markdown("**Phase 0 — Drill-in (current)**")
        st.markdown(
            "- Full LangGraph drill-in\n"
            "- Hybrid Owner-Earnings DCF + Multiple MC\n"
            "- 9-section synthesis report + PDF\n"
            "- 3 hand-written theses\n"
            "- Streamlit dashboard"
        )
    with cols[1], st.container(border=True):
        st.markdown("**Phase 1 — Personal MVP**")
        st.markdown(
            "- Notion memory layer\n"
            "- Bidirectional Telegram bot\n"
            "- Continuous Triage agent\n"
            "- DigitalOcean droplet deployment"
        )
    with cols[2], st.container(border=True):
        st.markdown("**Phase 2+ (deferred)**")
        st.markdown(
            "- Full Discovery agent (halo graph)\n"
            "- Bidirectional Notion sync\n"
            "- Pattern detection / multiplicity\n"
            "- Synthesis cycle-based re-trigger"
        )


def main() -> None:
    page_header(
        "Architecture",
        subtitle=(
            "What FINAQ is, how its parts fit together, and where to find the deep dives. "
            "Live data — agent cards read from your current `.env` and source modules."
        ),
    )

    # Top-of-page snapshot
    cols = st.columns(4)
    with cols[0]:
        freshness_card("Theses loaded", str(len(list((Path('theses')).glob('*.json')))))
    with cols[1]:
        ev_dir = Path("data_cache/eval/runs")
        n = len(list(ev_dir.glob("*.json"))) if ev_dir.exists() else 0
        freshness_card("Eval runs persisted", str(n))
    with cols[2]:
        demo_dir = Path("data_cache/demos")
        n = len(list(demo_dir.glob("*.json"))) if demo_dir.exists() else 0
        freshness_card("Cached demos", str(n))
    with cols[3]:
        freshness_card("LangSmith", os.environ.get("LANGSMITH_TRACING", "off"))

    section_divider()
    render_topology()
    section_divider()
    render_agent_cards()
    section_divider()
    render_data_sources()
    section_divider()
    render_test_tiers()
    section_divider()
    render_palette()
    section_divider()
    render_phase_scope()

    section_divider()
    st.caption(
        "Deep dives: `docs/ARCHITECTURE.md` (decision history), "
        "`docs/FINANCE_ASSUMPTIONS.md` (valuation methodology), "
        "`docs/POSTPONED.md` (deferred decisions with explicit triggers), "
        "`CLAUDE.md` (build spec)."
    )


main()
