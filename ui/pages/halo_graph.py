"""Halo graph — visualise a thesis's discovered relationship graph.

Auto-discovered by Streamlit from `ui/pages/`. Renders the Phase 2 Discovery
agent's halo graph (ARCHITECTURE §13): the universe of tickers around a thesis's
anchors and the supplier / customer / peer / competitor edges between them.

Two data sources, richest-first:
  1. The graph store (`data.graph` → state.db `graph_nodes` / `graph_edges`) —
     the full proposed+grounded graph a Discovery run persisted: confidence
     scores, grounded flags, and the filing / news evidence that corroborated
     each edge (including the pruned, ungrounded ones, kept for audit).
  2. Fallback — the thesis JSON's own `relationships` (already the grounded
     subset). Used for curated theses, or before any Discovery run has persisted
     a graph for this slug.

No new dependency: the diagram is drawn client-side with mermaid.js from a CDN —
reusing the Architecture page's topology-diagram embedding, with `htmlLabels`
enabled here so a node can show its ticker over its company name.
"""

from __future__ import annotations

# Bootstrap (see ui/app.py for explanation).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import html
import re
from dataclasses import dataclass, field

import streamlit as st
import streamlit.components.v1 as components

from data import graph
from ui.components import (
    BONE,
    EGGSHELL,
    INK,
    PARCHMENT,
    SAGE,
    TAUPE,
    WHITE,
    evidence_list,
    freshness_card,
    page_header,
    section_divider,
)
from utils.schemas import Thesis

st.set_page_config(page_title="FINAQ — Halo graph", page_icon="🕸️", layout="wide")

# Anchor to the repo root (like every sibling page), not CWD, so the page finds
# the theses no matter which directory `streamlit run ui/app.py` is launched from.
THESES_DIR = Path(__file__).resolve().parents[2] / "theses"

# The relationship type always rides in the edge label so all four types stay
# legible without adding colours — CLAUDE.md §13 keeps a single sage accent
# plus neutrals, so grounded/pruned is the only colour axis on edges.


# --- View model (uniform across both data sources) --------------------------


@dataclass
class VizNode:
    ticker: str
    name: str
    is_anchor: bool


@dataclass
class VizEdge:
    frm: str
    to: str
    type: str
    note: str
    confidence: float | None  # None when only the thesis file is available
    grounded: bool
    evidence: list[dict] = field(default_factory=list)


# --- Data loading ------------------------------------------------------------


def _list_theses() -> list[tuple[str, str]]:
    """(slug, display name) for every top-level thesis JSON, name-sorted.
    Skips the archive/ and backtest/ subdirectories and any unparseable file."""
    if not THESES_DIR.exists():
        return []
    out: list[tuple[str, str]] = []
    for path in THESES_DIR.glob("*.json"):
        try:
            thesis = Thesis.model_validate_json(path.read_text())
        except Exception:  # noqa: BLE001 — skip a malformed/legacy thesis file
            continue
        out.append((path.stem, thesis.name))
    return sorted(out, key=lambda t: t[1].lower())


def _load_thesis(slug: str) -> Thesis | None:
    try:
        return Thesis.model_validate_json((THESES_DIR / f"{slug}.json").read_text())
    except Exception:  # noqa: BLE001
        return None


def _build_model(slug: str, thesis: Thesis) -> tuple[list[VizNode], list[VizEdge], bool]:
    """Return (nodes, edges, from_graph_store).

    Prefer the graph store when it actually holds edges (richer: confidence +
    grounded + evidence, and it keeps the pruned edges). Fall back to the thesis
    file's grounded-only relationships otherwise — including the nodes-but-no-
    edges case (a discovery run that persisted a universe but grounded nothing),
    where the store adds no edges and the file may still carry curated ones."""
    anchors = set(thesis.anchor_tickers)
    g_nodes = graph.get_nodes(slug)
    g_edges = graph.get_edges(slug, grounded_only=False)

    if g_edges:
        name_by = {n.ticker: (n.name or n.ticker) for n in g_nodes}
        edges = [
            VizEdge(
                e.from_,
                e.to,
                e.type,
                e.note,
                float(e.confidence),
                bool(e.grounded),
                [ev.model_dump() for ev in e.evidence],
            )
            for e in g_edges
        ]
        # Node set = thesis universe ∪ every ticker any edge touches.
        tickers = list(thesis.universe)
        for e in edges:
            tickers.extend((e.frm, e.to))
        seen: set[str] = set()
        nodes = []
        for t in tickers:
            if t in seen:
                continue
            seen.add(t)
            nodes.append(VizNode(t, name_by.get(t, t), t in anchors))
        return nodes, edges, True

    # Fallback: thesis file only — relationships here are already grounded.
    nodes = [VizNode(t, t, t in anchors) for t in thesis.universe]
    edges = [VizEdge(r.from_, r.to, r.type, r.note, None, True, []) for r in thesis.relationships]
    return nodes, edges, False


# --- Mermaid generation ------------------------------------------------------


def _safe_id(ticker: str) -> str:
    """Mermaid node id: alphanumerics + underscore only (BRK.B → BRK_B)."""
    return re.sub(r"[^A-Za-z0-9]", "_", ticker) or "_"


def _node_id_map(nodes: list[VizNode]) -> dict[str, str]:
    """Map each distinct raw ticker to a UNIQUE mermaid node id.

    `_safe_id` alone is not injective — `BRK.B` and `BRK-B` both sanitise to
    `BRK_B`, and the store node set can carry both spellings (SEC uses `.`,
    yfinance uses `-`). Two `BRK_B[...]` declarations collapse into one mermaid
    node, silently dropping a company. Disambiguate collisions with a numeric
    suffix so every distinct ticker gets its own node; edges resolve endpoints
    through the same map."""
    used: set[str] = set()
    out: dict[str, str] = {}
    for n in nodes:
        base = _safe_id(n.ticker)
        sid = base
        k = 2
        while sid in used:
            sid = f"{base}_{k}"
            k += 1
        used.add(sid)
        out[n.ticker] = sid
    return out


def _safe_label(text: str, limit: int = 24) -> str:
    """Escape a company name for a quoted, htmlLabels mermaid node label.

    Ellipsise first (so the length cap counts visible characters, not entity
    expansions), then `html.escape` so `&`, `<`, `>`, `"` render as literal
    characters instead of breaking the label or the HTML span around it."""
    cleaned = (text or "").strip()
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 1] + "…"
    return html.escape(cleaned, quote=True)


def _mermaid(nodes: list[VizNode], edges: list[VizEdge], *, show_ungrounded: bool) -> str:
    lines = ["graph LR"]
    ids = _node_id_map(nodes)
    anchor_ids: list[str] = []
    member_ids: list[str] = []
    for n in nodes:
        sid = ids[n.ticker]
        name = _safe_label(n.name)
        # Two-line label (ticker over company name) via a plain <br/> — no
        # inline-styled span, which relies on the sanitiser keeping style attrs.
        label = f"{n.ticker}<br/>{name}" if name and name.upper() != n.ticker.upper() else n.ticker
        lines.append(f'    {sid}["{label}"]')
        (anchor_ids if n.is_anchor else member_ids).append(sid)

    shown = [e for e in edges if show_ungrounded or e.grounded]
    link_styles: list[str] = []
    for i, e in enumerate(shown):
        conf = f" · {e.confidence:.2f}" if e.confidence is not None else ""
        arrow = "-->" if e.grounded else "-.->"
        # Endpoints are always declared nodes (_build_model's invariant), so
        # resolve through the same id map — no _safe_id fallback that could
        # silently re-collide the ids _node_id_map just disambiguated.
        frm, to = ids[e.frm], ids[e.to]
        lines.append(f'    {frm} {arrow}|"{e.type}{conf}"| {to}')
        color = SAGE if e.grounded else TAUPE
        width = "2px" if e.grounded else "1.5px"
        link_styles.append(f"    linkStyle {i} stroke:{color},stroke-width:{width};")

    lines.append(f"    classDef anchor fill:{SAGE},stroke:{INK},color:{WHITE},font-weight:bold;")
    lines.append(f"    classDef member fill:{EGGSHELL},stroke:{TAUPE},color:{INK};")
    if anchor_ids:
        lines.append(f"    class {','.join(anchor_ids)} anchor;")
    if member_ids:
        lines.append(f"    class {','.join(member_ids)} member;")
    lines.extend(link_styles)
    return "\n".join(lines)


def _render_mermaid(code: str, n_nodes: int) -> None:
    """Embed the mermaid diagram client-side (mermaid.js from CDN — no new
    Python dep), themed with the CLAUDE.md §13 palette so it matches the rest
    of the dashboard. Height scales with node count."""
    height = max(420, min(150 + n_nodes * 62, 1100))
    doc = f"""
    <div style="background:{EGGSHELL}; padding:1rem; border-radius:6px;
        border:1px solid {TAUPE};">
      <pre class="mermaid">
{code}
      </pre>
    </div>
    <script type="module">
      import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
      mermaid.initialize({{
        startOnLoad: true,
        theme: "base",
        flowchart: {{ htmlLabels: true, curve: "basis" }},
        themeVariables: {{
          primaryColor: "{PARCHMENT}",
          primaryTextColor: "{INK}",
          primaryBorderColor: "{SAGE}",
          lineColor: "{SAGE}",
          secondaryColor: "{EGGSHELL}",
          tertiaryColor: "{WHITE}",
          fontSize: "14px",
        }},
      }});
    </script>
    """
    components.html(doc, height=height, scrolling=True)


def _legend() -> None:
    st.markdown(
        f"""
        <div style="display:flex; flex-wrap:wrap; gap:1.2rem; align-items:center;
            font-size:0.8rem; color:{INK}; margin-top:0.4rem;">
          <span><span style="display:inline-block; width:14px; height:14px;
            background:{SAGE}; border:1px solid {INK}; border-radius:3px;
            vertical-align:middle;"></span> anchor</span>
          <span><span style="display:inline-block; width:14px; height:14px;
            background:{EGGSHELL}; border:1px solid {TAUPE}; border-radius:3px;
            vertical-align:middle;"></span> universe member</span>
          <span><span style="display:inline-block; width:26px; height:0;
            border-top:2px solid {SAGE}; vertical-align:middle;"></span>
            grounded edge (corroborated)</span>
          <span><span style="display:inline-block; width:26px; height:0;
            border-top:2px dashed {TAUPE}; vertical-align:middle;"></span>
            pruned edge (ungrounded)</span>
          <span style="opacity:0.7;">edge label = relationship type · confidence</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- Edge table + evidence ---------------------------------------------------


def _confidence_bar(confidence: float | None) -> str:
    if confidence is None:
        return f"<span style='color:{INK}; opacity:0.6; font-size:0.8rem;'>from thesis file</span>"
    pct = int(round(max(0.0, min(1.0, confidence)) * 100))
    return (
        f"<div style='display:flex; align-items:center; gap:0.5rem;'>"
        f"<div style='background:{BONE}; border-radius:4px; height:10px; width:90px; "
        f"overflow:hidden;'><div style='background:{SAGE}; height:100%; width:{pct}%;'></div></div>"
        f"<span style='color:{INK}; font-size:0.8rem; font-weight:600;'>{confidence:.2f}</span></div>"
    )


def _edge_row(e: VizEdge) -> None:
    badge_fill, badge_text = (SAGE, WHITE) if e.grounded else (TAUPE, INK)
    badge = "grounded" if e.grounded else "pruned"
    cols = st.columns([2.4, 1.3, 1.6, 1.1])
    with cols[0]:
        st.markdown(
            f"<span style='font-weight:600; color:{INK};'>{e.frm} → {e.to}</span>",
            unsafe_allow_html=True,
        )
        if e.note:
            st.markdown(
                f"<span style='color:{INK}; opacity:0.7; font-size:0.82rem;'>{e.note}</span>",
                unsafe_allow_html=True,
            )
    with cols[1]:
        st.markdown(
            f"<span style='color:{INK}; font-size:0.85rem;'>{e.type}</span>",
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(_confidence_bar(e.confidence), unsafe_allow_html=True)
    with cols[3]:
        st.markdown(
            f"<span style='background:{badge_fill}; color:{badge_text}; padding:0.15rem 0.55rem; "
            f"border-radius:999px; font-size:0.7rem; font-weight:700; letter-spacing:0.04em; "
            f"text-transform:uppercase;'>{badge}</span>",
            unsafe_allow_html=True,
        )
    if e.evidence:
        with st.expander(f"Evidence for {e.frm} → {e.to} ({len(e.evidence)})"):
            evidence_list(e.evidence)


# --- Neighbours explorer (graph-store only) ----------------------------------


def _neighbours_explorer(slug: str, nodes: list[VizNode]) -> None:
    st.markdown("### Neighbourhood explorer")
    st.caption(
        "Recursive-CTE traversal over the stored graph (ARCHITECTURE §13.3): "
        "tickers reachable within N hops of a seed, following edges either way."
    )
    tickers = [n.ticker for n in nodes]
    cols = st.columns([2, 1, 1.4])
    with cols[0]:
        seed = st.selectbox("Seed ticker", tickers, key="halo_seed")
    with cols[1]:
        hops = st.slider("Hops", 1, graph.MAX_HOPS, 1, key="halo_hops")
    with cols[2]:
        grounded_only = st.toggle("Grounded edges only", value=True, key="halo_nb_grounded")
    reached = graph.neighbors(seed, hops=hops, thesis_slug=slug, grounded_only=grounded_only)
    if not reached:
        st.caption(f"No tickers within {hops} hop(s) of {seed} on this graph.")
        return
    chips = "".join(
        f"<span style='background:{EGGSHELL}; border:1px solid {TAUPE}; color:{INK}; "
        f"padding:0.2rem 0.7rem; border-radius:999px; font-size:0.85rem; font-weight:600; "
        f"margin:0.2rem;'>{t}</span>"
        for t in reached
    )
    st.markdown(
        f"<div style='margin-top:0.4rem;'><span style='color:{INK}; opacity:0.7; "
        f"font-size:0.85rem;'>{len(reached)} reachable from <b>{seed}</b>:</span><br/>{chips}</div>",
        unsafe_allow_html=True,
    )


# --- Page render -------------------------------------------------------------


def _empty_state() -> None:
    page_header(
        "Halo graph",
        subtitle="Visualise the relationship graph the Discovery agent builds around a thesis.",
    )
    st.info(
        "No theses found in `theses/`. Run the Discovery agent to build one:\n\n"
        "```bash\npython -m scripts.run_discovery "
        '--topic "AI datacenter power, energy and chips" --ingest\n```'
    )


def main() -> None:
    theses = _list_theses()
    if not theses:
        _empty_state()
        return

    page_header(
        "Halo graph",
        subtitle=(
            "The universe of tickers around a thesis's anchors and the "
            "supplier / customer / peer / competitor edges between them. Discovery "
            "proposes edges freely, then grounds each against real filings + news "
            "(“B grounds A”) — only corroborated edges enter the thesis."
        ),
    )

    # Selector — mark theses whose stored graph has edges. Keyed off edges (not
    # nodes) to match `_build_model`'s source decision: a slug with persisted
    # nodes but no edges renders from the thesis file, so it must not show the ●.
    has_graph = {slug: bool(graph.get_edges(slug)) for slug, _ in theses}
    labels = [f"{'● ' if has_graph[slug] else '○ '}{name}  ·  {slug}" for slug, name in theses]
    default_idx = next((i for i, (slug, _) in enumerate(theses) if has_graph[slug]), 0)
    choice = st.selectbox(
        "Thesis",
        range(len(theses)),
        index=default_idx,
        format_func=lambda i: labels[i],
        help="● has a persisted discovery graph (confidence + evidence)  ·  "
        "○ curated / not yet discovered (edges read from the thesis file)",
    )
    slug, _name = theses[choice]
    thesis = _load_thesis(slug)
    if thesis is None:
        st.error(f"Could not load thesis `{slug}`.")
        return

    nodes, edges, from_store = _build_model(slug, thesis)
    grounded = [e for e in edges if e.grounded]

    # Stats strip.
    section_divider()
    cols = st.columns(4)
    with cols[0]:
        freshness_card("Universe", str(len(nodes)))
    with cols[1]:
        freshness_card("Anchors", ", ".join(thesis.anchor_tickers) or "—")
    with cols[2]:
        if from_store:
            freshness_card("Edges grounded", f"{len(grounded)} / {len(edges)}")
        else:
            freshness_card("Relationships", str(len(edges)))
    with cols[3]:
        freshness_card(
            "Source",
            "discovery graph" if from_store else "thesis file",
            note="state.db" if from_store else "grounded subset",
        )

    st.markdown(
        f"<p style='color:{INK}; opacity:0.85; margin:0.6rem 0 0.2rem;'>{thesis.summary}</p>",
        unsafe_allow_html=True,
    )

    # Diagram.
    section_divider()
    show_ungrounded = False
    if from_store and len(edges) > len(grounded):
        show_ungrounded = st.toggle(
            f"Show {len(edges) - len(grounded)} pruned (ungrounded) edges",
            value=False,
            help="Ungrounded edges were proposed by the LLM but not corroborated "
            "against filings or news, so they were kept out of the thesis.",
        )

    if not edges:
        st.caption("No relationships surfaced for this thesis yet — showing the universe only.")
    code = _mermaid(nodes, edges, show_ungrounded=show_ungrounded)
    _render_mermaid(code, len(nodes))
    _legend()
    with st.expander("Show mermaid source"):
        st.code(code, language="text")

    # Edge table.
    section_divider()
    st.markdown("### Relationships")
    table_edges = edges if show_ungrounded else grounded
    if not table_edges:
        st.caption("No grounded relationships to list.")
    else:
        st.caption(
            "Highest-confidence first. Expand a row to see the filing / news "
            "citations that corroborated the edge."
            if from_store
            else "Relationships read from the thesis file (already the grounded subset)."
        )
        for e in table_edges:
            _edge_row(e)

    # Neighbourhood explorer — only meaningful against the stored graph.
    if from_store and grounded:
        section_divider()
        _neighbours_explorer(slug, nodes)

    section_divider()
    st.caption(
        "How the graph is built: `agents/discovery.py` (propose-then-verify) · "
        "stored in `data/graph.py` (state.db) · design in `docs/ARCHITECTURE.md` §13."
    )


main()
