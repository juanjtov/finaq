"""Reusable Streamlit widgets for the FINAQ dashboard.

Every dashboard page imports from here so the visual language stays
identical across surfaces: KPI tables, evidence renderers, MC chart
wrapper, scenario summary card, watchlist renderer, freshness card,
confidence badge.

CLAUDE.md §13 palette is the source of truth for colour. Widget styling
uses Streamlit's native primitives (`st.metric`, `st.columns`,
`st.markdown` with embedded HTML/CSS) — no custom React component.
"""

from __future__ import annotations

from collections.abc import Iterable

import streamlit as st

from utils import humanize_amount
from utils.charts import mc_histogram

# --- Palette (mirrored from utils/charts.py / utils/pdf_export.py) ----------

SAGE = "#2D4F3A"
PARCHMENT = "#F4ECDC"
WHITE = "#FFFFFF"
EGGSHELL = "#FBF5E8"
TAUPE = "#E0D5C2"
BONE = "#EDE5D5"
INK = "#1A1611"


# --- Confidence badge -------------------------------------------------------


def confidence_badge(confidence: str) -> None:
    """Render a small palette-coded confidence pill.
    high → sage on white text; medium → taupe on ink; low → bone on ink."""
    confidence = (confidence or "medium").lower()
    fill, text = {
        "high": (SAGE, WHITE),
        "medium": (TAUPE, INK),
        "low": (BONE, INK),
    }.get(confidence, (TAUPE, INK))
    st.markdown(
        f"""
        <span style="display:inline-block; padding:0.25rem 0.75rem;
            border-radius:999px; background:{fill}; color:{text};
            font-weight:600; font-size:0.85rem; letter-spacing:0.04em;">
            CONFIDENCE: {confidence.upper()}
        </span>
        """,
        unsafe_allow_html=True,
    )


# --- KPI table --------------------------------------------------------------

# Same display set + formatting as utils/pdf_export._COVER_KPIS. Kept here
# so the dashboard and PDF render identically when both have access to
# `state.fundamentals.kpis`. The formatter strings are interpreted by
# `_format_kpi_value` below; "humanize_$" / "humanize" are special tokens
# that delegate to utils.humanize_amount for B/M/K compaction.
_KPI_ROWS: tuple[tuple[str, str, str], ...] = (
    ("current_price", "Current price", "${:,.2f}"),
    ("revenue_latest", "Latest revenue", "humanize_$"),
    ("operating_margin_5yr_avg", "Op. margin (5y avg)", "{:.1%}"),
    ("fcf_yield", "FCF yield", "{:.2f}%"),
    ("pe_trailing", "P/E (trailing)", "{:.1f}x"),
    ("revenue_5y_cagr", "Revenue CAGR (5y)", "{:.1%}"),
    ("net_cash", "Net cash", "humanize_$"),
    ("market_cap", "Market cap", "humanize_$"),
    ("shares_outstanding", "Shares outstanding", "humanize"),
)


def _format_kpi_value(raw: object, fmt: str) -> str:
    if raw is None:
        return "—"
    if fmt == "humanize_$":
        return humanize_amount(raw, prefix="$")
    if fmt == "humanize":
        return humanize_amount(raw, prefix="")
    try:
        value = float(raw)
        if value != value:
            return "—"
        return fmt.format(value)
    except (TypeError, ValueError):
        return str(raw)


def kpi_grid(kpis: dict, columns: int = 4) -> None:
    """Render KPI cards as Streamlit `st.metric` widgets in a column grid.
    KPIs missing from the dict are silently skipped."""
    if not kpis:
        st.caption("No KPI data available.")
        return
    rows: list[tuple[str, str]] = []
    for key, label, fmt in _KPI_ROWS:
        if key not in kpis or kpis.get(key) is None:
            continue
        rows.append((label, _format_kpi_value(kpis[key], fmt)))
    if not rows:
        st.caption("No KPI data available.")
        return
    cols = st.columns(columns)
    for i, (label, value) in enumerate(rows):
        cols[i % columns].metric(label, value)


# --- MC chart wrapper -------------------------------------------------------


def mc_chart(samples, current_price: float | None, caption: str | None = None) -> None:
    """Render the MC histogram in Streamlit. Closes the matplotlib figure
    after rendering so the dashboard doesn't leak figures across reruns."""
    fig = mc_histogram(samples, current_price=current_price)
    st.pyplot(fig, use_container_width=True)
    if caption:
        st.caption(caption)
    import matplotlib.pyplot as plt

    plt.close(fig)


# --- Scenario summary card --------------------------------------------------


def scenario_card(mc: dict) -> None:
    """Render Bull/Base/Bear band ranges as a 3-column card. Reads percentile
    bands from `state.monte_carlo.dcf` (DCF model is the primary spine)."""
    dcf = (mc or {}).get("dcf") or {}
    if not dcf:
        st.caption("No Monte Carlo distribution available.")
        return
    cols = st.columns(3)
    bands = (
        ("Bear (P10–P25)", dcf.get("p10"), dcf.get("p25"), TAUPE),
        ("Base (P25–P75)", dcf.get("p25"), dcf.get("p75"), SAGE),
        ("Bull (P75–P90)", dcf.get("p75"), dcf.get("p90"), TAUPE),
    )
    for col, (label, lo, hi, color) in zip(cols, bands, strict=False):
        if lo is None or hi is None:
            col.markdown(f"**{label}**: —")
            continue
        col.markdown(
            f"""
            <div style="border-left:4px solid {color}; padding:0.4rem 0.75rem;
                background:{EGGSHELL}; border-radius:4px; min-height:4rem;">
                <div style="font-size:0.75rem; color:{INK}; letter-spacing:0.04em;
                    text-transform:uppercase; font-weight:600;">{label}</div>
                <div style="font-size:1.1rem; color:{INK}; font-weight:600; margin-top:0.2rem;">
                    ${lo:,.0f} – ${hi:,.0f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --- Evidence list ----------------------------------------------------------


def evidence_list(evidence: Iterable[dict], heading: str | None = None) -> None:
    """Render a flat list of evidence items as palette-coded bullet rows
    with clickable URLs / accession numbers. Used inside per-agent expanders."""
    items = list(evidence or [])
    if heading:
        st.caption(heading)
    if not items:
        st.caption("No evidence items.")
        return
    for ev in items:
        source = ev.get("source", "?")
        url = ev.get("url")
        accession = ev.get("accession")
        item = ev.get("item")
        excerpt = (ev.get("excerpt") or "").strip()
        as_of = ev.get("as_of")
        # Compose the lead line (source · item · date)
        lead_parts = [f"**{source}**"]
        if item:
            lead_parts.append(item)
        if as_of:
            lead_parts.append(as_of)
        lead = " · ".join(lead_parts)
        # Body line
        body_parts: list[str] = []
        if url:
            body_parts.append(f"[link]({url})")
        if accession:
            body_parts.append(f"`{accession}`")
        if excerpt:
            body_parts.append(f"_{excerpt[:200]}_")
        body = " — ".join(body_parts) if body_parts else ""
        st.markdown(f"- {lead} — {body}" if body else f"- {lead}")


# --- Watchlist renderer -----------------------------------------------------


def watchlist_card(watchlist: list[str]) -> None:
    """Render the watchlist items as bullet cards with the agent suffix
    rendered as a colour-coded chip (filings = sage, news = taupe,
    fundamentals = bone, risk = ink)."""
    if not watchlist:
        st.caption("No watchlist items.")
        return
    chip_colors = {
        "filings": SAGE,
        "news": TAUPE,
        "fundamentals": BONE,
        "risk": INK,
        "synthesis": SAGE,
    }
    for item in watchlist:
        # Detect the trailing agent suffix `(<agent>)`
        agent = "?"
        text = item
        for name in ("filings", "news", "fundamentals", "risk", "synthesis"):
            suffix = f"({name})"
            if item.rstrip().endswith(suffix):
                agent = name
                text = item.rstrip()[: -len(suffix)].rstrip()
                break
        chip = chip_colors.get(agent, INK)
        text_color = WHITE if agent in ("filings", "synthesis", "risk") else INK
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:0.6rem; padding:0.4rem 0;
                border-bottom:1px solid {BONE};">
                <span style="background:{chip}; color:{text_color};
                    padding:0.15rem 0.6rem; border-radius:999px; font-size:0.7rem;
                    font-weight:600; letter-spacing:0.04em; text-transform:uppercase;
                    min-width:6rem; text-align:center;">{agent}</span>
                <span style="color:{INK}; font-size:0.95rem;">{text}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --- Freshness card ---------------------------------------------------------


def freshness_card(label: str, timestamp: str | None, note: str = "") -> None:
    """A small data-source freshness card used on Mission Control and (in
    Step 5z onward) tied to `data_cache/state.db`."""
    text = timestamp or "—"
    st.markdown(
        f"""
        <div style="border:1px solid {TAUPE}; border-radius:6px; padding:0.6rem 0.9rem;
            background:{EGGSHELL}; min-height:4rem;">
            <div style="font-size:0.7rem; color:{INK}; letter-spacing:0.05em;
                text-transform:uppercase; font-weight:600;">{label}</div>
            <div style="color:{INK}; font-size:0.95rem; font-weight:600; margin-top:0.2rem;">
                {text}
            </div>
            <div style="color:{INK}; opacity:0.65; font-size:0.8rem; margin-top:0.2rem;">
                {note}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- Page header ------------------------------------------------------------


def page_header(title: str, subtitle: str | None = None) -> None:
    """Consistent page header — sage h1 + ink subtitle + taupe divider."""
    st.markdown(
        f"""
        <div style="margin-bottom:1.0rem;">
            <h1 style="color:{SAGE}; margin-bottom:0.2rem; font-weight:700;
                line-height:1.2;">{title}</h1>
            {f'<p style="color:{INK}; opacity:0.75; margin:0;">{subtitle}</p>' if subtitle else ''}
            <hr style="border:none; height:2px; background:{TAUPE}; margin:0.8rem 0 1.4rem 0;" />
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- Valuation badge (current price vs MC distribution) --------------------


def valuation_badge(current_price: float | None, mc: dict | None) -> None:
    """Visualises the gap between current price and the MC P50/P25/P75 bands
    as a coloured pill. The verdict is the first thing the user reads."""
    if not current_price or not mc:
        st.caption("Valuation badge unavailable — MC distribution missing.")
        return
    dcf = (mc.get("dcf") or {}) if isinstance(mc, dict) else {}
    p25, p50, p75, p10, p90 = (
        dcf.get("p25"),
        dcf.get("p50"),
        dcf.get("p75"),
        dcf.get("p10"),
        dcf.get("p90"),
    )
    if not all(v is not None for v in (p10, p25, p50, p75, p90)):
        st.caption("Valuation badge unavailable — distribution incomplete.")
        return

    cp = float(current_price)
    if cp < p10:
        label, fg, bg = "DEEP VALUE", WHITE, SAGE
        comment = f"current price ${cp:,.0f} is below the 10th percentile (${p10:,.0f})"
    elif cp < p25:
        label, fg, bg = "UNDERVALUED", WHITE, SAGE
        comment = f"current price ${cp:,.0f} sits below the 25th-percentile DCF estimate (${p25:,.0f})"
    elif cp <= p75:
        label, fg, bg = "FAIRLY VALUED", INK, TAUPE
        comment = (
            f"current price ${cp:,.0f} is inside the central P25-P75 band "
            f"(${p25:,.0f} - ${p75:,.0f}); midpoint ${p50:,.0f}"
        )
    elif cp <= p90:
        label, fg, bg = "RICH", INK, BONE
        comment = (
            f"current price ${cp:,.0f} is above the 75th-percentile DCF "
            f"estimate (${p75:,.0f}) — limited upside"
        )
    else:
        label, fg, bg = "OVERVALUED", WHITE, INK
        comment = (
            f"current price ${cp:,.0f} is above the 90th-percentile DCF "
            f"estimate (${p90:,.0f}) — significantly above the model's view"
        )
    upside_to_p50 = (p50 - cp) / cp * 100 if cp else 0.0
    st.markdown(
        f"""
        <div style="display:flex; flex-direction:column; gap:0.4rem;">
            <span style="display:inline-block; padding:0.4rem 1.1rem;
                border-radius:999px; background:{bg}; color:{fg};
                font-weight:700; font-size:1.1rem; letter-spacing:0.06em;
                width:fit-content;">{label}</span>
            <div style="color:{INK}; font-size:0.95rem;">
                Model midpoint <b>${p50:,.0f}</b>
                ({upside_to_p50:+.1f}% vs current).
            </div>
            <div style="color:{INK}; opacity:0.7; font-size:0.85rem;">
                {comment}.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- Hero strip -------------------------------------------------------------


def hero_strip(
    ticker: str,
    thesis_name: str,
    current_price: float | None,
    mc: dict | None,
    confidence: str,
) -> None:
    """Top-of-dashboard summary strip. Pulls the user into the first frame
    answer (price + valuation verdict + confidence) before they scroll."""
    cols = st.columns([3, 4, 2])
    with cols[0]:
        st.markdown(
            f"""
            <div style="display:flex; flex-direction:column; gap:0.3rem;">
                <div style="font-size:0.8rem; color:{INK}; opacity:0.65;
                    text-transform:uppercase; letter-spacing:0.06em;">
                    {thesis_name}
                </div>
                <div style="font-size:2.6rem; line-height:1.1; color:{SAGE};
                    font-weight:700;">{ticker}</div>
                <div style="font-size:1.6rem; color:{INK}; font-weight:600;">
                    {f"${float(current_price):,.2f}" if current_price else "—"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[1]:
        valuation_badge(current_price, mc)
    with cols[2]:
        confidence_badge(confidence)


# --- Risk gauge -------------------------------------------------------------


def risk_gauge(level: str | None, score_0_to_10: int | None) -> None:
    """Visual risk dial: pill with categorical level + horizontal bar
    showing the 0-10 score. Palette-consistent (sage when low, taupe when
    elevated, ink when critical — no red, per CLAUDE.md §13)."""
    level = (level or "MODERATE").upper()
    score = max(0, min(10, int(score_0_to_10 or 0)))
    pill_fill = {
        "LOW": SAGE,
        "MODERATE": SAGE,
        "ELEVATED": TAUPE,
        "HIGH": INK,
        "CRITICAL": INK,
    }.get(level, TAUPE)
    pill_text = WHITE if pill_fill in (SAGE, INK) else INK
    bar_fill = pill_fill
    pct = score * 10  # bar width as %
    st.markdown(
        f"""
        <div style="display:flex; flex-direction:column; gap:0.5rem;">
            <div style="display:flex; align-items:center; gap:0.7rem;">
                <span style="background:{pill_fill}; color:{pill_text};
                    padding:0.3rem 0.9rem; border-radius:999px;
                    font-size:0.85rem; font-weight:700; letter-spacing:0.06em;">
                    {level}
                </span>
                <span style="color:{INK}; font-weight:600;">
                    Score: {score}/10
                </span>
            </div>
            <div style="background:{BONE}; border-radius:6px; height:14px;
                width:100%; overflow:hidden;">
                <div style="background:{bar_fill}; height:100%;
                    width:{pct}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- News split (catalysts vs concerns) ------------------------------------


def news_split(catalysts: list[dict], concerns: list[dict]) -> None:
    """Two-column card view of bull catalysts (left) vs bear concerns (right).
    Each item rendered as a card with title, date, and clickable URL."""
    cols = st.columns(2)
    with cols[0]:
        st.markdown(
            f"<h4 style='color:{SAGE}; margin-bottom:0.4rem;'>↗ Catalysts</h4>",
            unsafe_allow_html=True,
        )
        if not catalysts:
            st.caption("No catalysts surfaced.")
        for c in catalysts[:6]:
            _render_news_card(c, accent=SAGE)
    with cols[1]:
        st.markdown(
            f"<h4 style='color:{INK}; margin-bottom:0.4rem;'>↘ Concerns</h4>",
            unsafe_allow_html=True,
        )
        if not concerns:
            st.caption("No concerns surfaced.")
        for c in concerns[:6]:
            _render_news_card(c, accent=INK)


def _render_news_card(item: dict, accent: str) -> None:
    title = item.get("title", "(untitled)")
    summary = (item.get("summary") or "").strip()
    url = item.get("url") or ""
    as_of = item.get("as_of") or "?"
    sentiment = item.get("sentiment") or ""
    st.markdown(
        f"""
        <div style="background:{EGGSHELL}; border-left:4px solid {accent};
            border-radius:4px; padding:0.7rem 1rem; margin-bottom:0.5rem;">
            <div style="font-weight:600; color:{INK}; line-height:1.3;
                margin-bottom:0.3rem;">
                <a href="{url}" target="_blank" style="color:{INK}; text-decoration:none;">
                    {title}
                </a>
            </div>
            <div style="font-size:0.78rem; color:{INK}; opacity:0.7;
                letter-spacing:0.04em;">
                {as_of} · {sentiment}
            </div>
            {f'<div style="font-size:0.85rem; color:{INK}; margin-top:0.3rem;">{summary[:180]}</div>' if summary else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- Top risks chips --------------------------------------------------------


def top_risks_chips(top_risks: list[dict], limit: int = 5) -> None:
    """Numbered cards for the Risk agent's top_risks list, severity-coded."""
    if not top_risks:
        st.caption("No top risks recorded.")
        return
    severity_color = {1: BONE, 2: BONE, 3: TAUPE, 4: TAUPE, 5: INK}
    for i, r in enumerate(top_risks[:limit], start=1):
        severity = int(r.get("severity") or 3)
        accent = severity_color.get(severity, TAUPE)
        text_color = WHITE if accent == INK else INK
        st.markdown(
            f"""
            <div style="background:{EGGSHELL}; border-left:5px solid {accent};
                padding:0.6rem 1rem; border-radius:4px; margin-bottom:0.5rem;">
                <div style="display:flex; align-items:baseline; gap:0.6rem;">
                    <span style="background:{accent}; color:{text_color};
                        padding:0.15rem 0.55rem; border-radius:4px;
                        font-size:0.7rem; font-weight:700; letter-spacing:0.04em;">
                        #{i} · SEV {severity}
                    </span>
                    <span style="color:{INK}; font-weight:600;">
                        {r.get('title', '')}
                    </span>
                </div>
                <div style="color:{INK}; opacity:0.85; font-size:0.88rem;
                    margin-top:0.3rem; line-height:1.4;">
                    {r.get('explanation', '')}
                </div>
                <div style="color:{INK}; opacity:0.55; font-size:0.75rem;
                    margin-top:0.3rem; letter-spacing:0.04em;">
                    sources: {', '.join(r.get('sources') or []) or '—'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --- Section divider --------------------------------------------------------


def section_divider() -> None:
    st.markdown(
        f'<hr style="border:none; height:1px; background:{BONE}; margin:1.2rem 0;" />',
        unsafe_allow_html=True,
    )


# --- Per-agent expander -----------------------------------------------------


def agent_expander(name: str, payload: dict, evidence_label: str = "Evidence") -> None:
    """Render one agent's full structured output in a Streamlit expander.
    Generic shape (works for fundamentals/filings/news/risk):
      - summary string
      - structured fields (auto-detected: kpis, projections, risk_themes,
        catalysts, concerns, top_risks, etc.)
      - evidence list at the bottom"""
    with st.expander(f"{name.capitalize()} — full agent output"):
        if not payload:
            st.caption("Agent produced no output for this run.")
            return
        if payload.get("summary"):
            st.markdown("**Summary**")
            st.write(payload["summary"])
        # Per-agent extra fields — render only what's present
        if "kpis" in payload and payload["kpis"]:
            st.markdown("**KPIs**")
            st.json(payload["kpis"])
        if "projections" in payload and payload["projections"]:
            st.markdown("**Projections (MC inputs)**")
            st.json(payload["projections"])
        if "risk_themes" in payload and payload["risk_themes"]:
            st.markdown("**Risk themes**")
            for r in payload["risk_themes"]:
                st.markdown(f"- {r}")
        if "mdna_quotes" in payload and payload["mdna_quotes"]:
            st.markdown("**MD&A quotes**")
            for q in payload["mdna_quotes"]:
                st.markdown(
                    f"> {q.get('text', '')}\n\n"
                    f"_{q.get('item', '?')} · {q.get('accession', '?')}_"
                )
        if "catalysts" in payload and payload["catalysts"]:
            st.markdown("**Catalysts (bull/neutral)**")
            for c in payload["catalysts"]:
                st.markdown(
                    f"- **[{c.get('sentiment', '?')}]** "
                    f"[{c.get('title', '')}]({c.get('url', '#')}) "
                    f"({c.get('as_of', '?')})"
                )
        if "concerns" in payload and payload["concerns"]:
            st.markdown("**Concerns (bear/neutral)**")
            for c in payload["concerns"]:
                st.markdown(
                    f"- **[{c.get('sentiment', '?')}]** "
                    f"[{c.get('title', '')}]({c.get('url', '#')}) "
                    f"({c.get('as_of', '?')})"
                )
        if "top_risks" in payload and payload["top_risks"]:
            st.markdown("**Top risks**")
            for i, r in enumerate(payload["top_risks"], start=1):
                st.markdown(
                    f"{i}. **{r.get('title', '')}** "
                    f"(severity {r.get('severity', '?')}) — "
                    f"{r.get('explanation', '')} "
                    f"_sources: {', '.join(r.get('sources', []))}_"
                )
        if "convergent_signals" in payload and payload["convergent_signals"]:
            st.markdown("**Convergent signals**")
            for s in payload["convergent_signals"]:
                st.markdown(
                    f"- **{s.get('theme', '')}** "
                    f"_(in {', '.join(s.get('sources', []))})_ — "
                    f"{s.get('explanation', '')}"
                )
        if "threshold_breaches" in payload and payload["threshold_breaches"]:
            st.markdown("**Threshold breaches**")
            for b in payload["threshold_breaches"]:
                st.markdown(
                    f"- `{b.get('signal', '')}` "
                    f"{b.get('operator', '')} {b.get('threshold_value', '')} "
                    f"(observed = {b.get('observed_value', '?')}, "
                    f"source: {b.get('source', '?')})"
                )
        if "level" in payload:
            st.markdown(
                f"**Level:** {payload.get('level', '?')} (score "
                f"{payload.get('score_0_to_10', '?')} / 10)"
            )
        ev = payload.get("evidence")
        if ev:
            st.markdown(f"**{evidence_label}**")
            evidence_list(ev)
        errs = payload.get("errors")
        if errs:
            st.warning("Agent reported errors: " + "; ".join(str(e) for e in errs))
