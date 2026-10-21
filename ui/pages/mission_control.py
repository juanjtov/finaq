"""Mission Control — Live | Runs | Health (Mission Control v2, Phase 2).

Three tabs over one telemetry store, `data_cache/state.db` (schema v9):

  - **Live** — an `st.fragment(run_every="1s")` closes stale rows, reads
    `data.state.live_snapshot()` and feeds it to the embedded control-room
    component (`ui/mc_live.html`, mockup C with mockup B's waterfall). The
    snapshot is hashed without `now`; the component HTML is rebuilt only when
    that hash changes and the previous string is re-emitted otherwise, so
    Streamlit keeps the iframe mounted and the page animates client-side
    between polls (docs/MISSION_CONTROL_PLAN.md §3.3, §6).
  - **Runs** — the former Run Inspector: run picker (newest first; `?run_id=`
    and the `inspect_run_id` session key preselect), header, waterfall, agent
    cards, the LLM call trace with the FULL system / user / output text from
    `run_snapshot()`, errors, saved payloads, CIO lineage, and a Replay of the
    run through the same component with a scrubber.
  - **Health** — the pre-v2 panels, unchanged: freshness, filings manifest,
    runs table, daily cost, CIO actions / cycles / model performance, errors,
    Notion, eval runs, cached demos.

`ui/pages/run_inspector.py` redirects here keeping `run_id`.
"""

from __future__ import annotations

# Bootstrap (see ui/app.py for explanation).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hashlib
import html
import json
import os
from datetime import UTC, datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from data import state as state_db
from ui.components import (
    EGGSHELL,
    INK,
    SAGE,
    TAUPE,
    freshness_card,
    mc_tokens,
    md_safe,
    page_header,
    section_divider,
)
from utils import logger

st.set_page_config(page_title="FINAQ — Mission Control", page_icon="🛰️", layout="wide")

EVAL_DIR = Path(__file__).parents[2] / "data_cache" / "eval" / "runs"
DEMO_DIR = Path(__file__).parents[2] / "data_cache" / "demos"
EDGAR_DIR = Path(__file__).parents[2] / "data_cache" / "edgar"
YFIN_DIR = Path(__file__).parents[2] / "data_cache" / "yfin"
MC_TEMPLATE = Path(__file__).resolve().parents[1] / "mc_live.html"

# Component iframe heights (px). The component lays itself out for a ~1100 px
# wide container; `scrolling=True` keeps anything taller reachable.
LIVE_HEIGHT = 1100
REPLAY_HEIGHT = 1000

# Semantic status colors — deliberately outside the §13 brand palette so a
# failed run is unmissable against the warm-neutral surfaces.
FAIL = "#A33D2E"   # brick — failed nodes / failed run border
WARN = "#9A6B1F"   # amber — degraded (completed with soft failures)

# Canonical node order for timeline rendering (graph topology order).
NODE_ORDER = (
    "load_thesis", "fundamentals", "filings", "news",
    "risk", "monte_carlo", "synthesis",
)


# --- Live tab ---------------------------------------------------------------


def snapshot_hash(snapshot: dict) -> str:
    """Content hash of a snapshot WITHOUT `now` — the key that decides whether
    the component HTML is rebuilt (plan §3.3)."""
    body = {k: v for k, v in snapshot.items() if k != "now"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def live_html(snapshot: dict, *, mode: str = "live", theme: str | None = None) -> str:
    """Render `ui/mc_live.html` for one snapshot: palette and font tokens from
    `mc_tokens()`, `{{mode}}` (live | replay), `{{theme}}`, and the snapshot
    as a JSON data block. `<` is escaped inside the JSON so a prompt that
    contains `</script>` cannot break out of the block. Pure: the same
    snapshot and settings give the same bytes."""
    tokens = mc_tokens(theme or os.getenv("FINAQ_MC_THEME", "day"))
    payload = {**snapshot, "mode": mode}
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    tokens["snapshot_json"] = data.replace("<", "\\u003c")
    tokens["mode"] = mode
    out = MC_TEMPLATE.read_text(encoding="utf-8")
    for key, value in tokens.items():
        out = out.replace("{{" + key + "}}", value)
    return out


# One entry: the last snapshot hash and the HTML built for it. Module-level
# rather than session_state so every browser tab shares one build per change
# and the fragment body can be driven directly in tests. Re-emitting the same
# string is what keeps Streamlit from re-mounting the iframe.
_LIVE_CACHE: dict[str, str | None] = {"hash": None, "html": None}


def render_live_panel() -> None:
    """Body of the Live fragment: close stale rows (the sweeper is a writer and
    the caller's job — `live_snapshot()` is a pure read), read the snapshot,
    rebuild the HTML only when its hash changed, and emit exactly ONE
    `components.html`."""
    if Path(state_db.DB_PATH).exists():  # never create the file from a poll
        try:
            state_db.sweep_stale_runs()
        except Exception as exc:  # a locked file must not blank the live view
            logger.warning(f"[mission-control] stale sweep skipped: {exc}")
    snapshot = state_db.live_snapshot()
    digest = snapshot_hash(snapshot)
    if _LIVE_CACHE["hash"] != digest or _LIVE_CACHE["html"] is None:
        _LIVE_CACHE["html"] = live_html(snapshot)
        _LIVE_CACHE["hash"] = digest
    components.html(_LIVE_CACHE["html"], height=LIVE_HEIGHT, scrolling=True)


# `run_every` is inert under AppTest; tests call `render_live_panel()` directly.
_live_fragment = st.fragment(run_every=f"{state_db.LIVE_POLL_S}s")(render_live_panel)


def replay_snapshot(run_id: str) -> dict | None:
    """`run_snapshot(run_id)` wrapped in the live snapshot's shape so the same
    component replays it: `now` is the run's end, `cio` carries the cycle
    behind the run (steps + decisions) when there is one, `medians` feed the
    progress rings. None when the run is unknown."""
    run = state_db.run_snapshot(run_id)
    if run is None:
        return None
    lineage = run.get("cio") or {}
    cycle = state_db.get_cio_run(str(lineage.get("cio_run_id") or "")) if lineage else None
    cio = []
    if cycle:
        actions = state_db.recent_cio_actions(limit=200, cio_run_id=cycle["run_id"])
        cio = [
            {
                **cycle,
                "steps": state_db.spans_for("cio_run", cycle["run_id"]),
                "actions": list(reversed(actions)),  # chronological
            }
        ]
    ends = [n.get("ended_at") for n in run.get("nodes") or [] if n.get("ended_at")]
    now = run.get("ended_at") or (max(ends) if ends else run.get("started_at"))
    return {
        "now": now,
        "poll_s": 0,
        "vitals": {},
        "medians": state_db.node_medians(),
        "cio": cio,
        "runs": [run],
        "recent": [],
    }


def render_replay(run_id: str) -> None:
    snapshot = replay_snapshot(run_id)
    if snapshot is None:
        st.caption("Nothing to replay — this run is not in `state.db`.")
        return
    components.html(live_html(snapshot, mode="replay"), height=REPLAY_HEIGHT, scrolling=True)


# --- Runs tab (the former Run Inspector) ------------------------------------

_RI_CSS = """
<style>
.ri-stats { display: grid; grid-template-columns: repeat(5, 1fr); border: 1px solid #E0D5C2;
  border-radius: 10px; background: #FFFFFF; overflow: hidden; margin: 2px 0 6px; }
.ri-stats .cell { padding: 12px 16px; border-right: 1px solid #EDE5D5; }
.ri-stats .cell:last-child { border-right: none; }
.ri-stats .lbl { font-size: 10.5px; letter-spacing: 0.07em; text-transform: uppercase; color: #6B6152; }
.ri-stats .val { font: 500 19px var(--ri-mono, ui-monospace, "SF Mono", Menlo, monospace);
  color: #1A1611; margin-top: 2px; font-variant-numeric: tabular-nums; }
.ri-cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 4px 0 6px; }
.ri-card { background: #FFFFFF; border: 1px solid #E0D5C2; border-radius: 10px; padding: 13px 15px; }
.ri-card.failed { border-color: #DBB3A8; background: #F6E3DD; }
.ri-card.running { border-color: #2D4F3A; }
.ri-card .top { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }
.ri-card .nm { font: 600 13px var(--ri-mono, ui-monospace, "SF Mono", Menlo, monospace); color: #1A1611; }
.ri-card .pill { font: 600 10px sans-serif; letter-spacing: 0.04em; text-transform: uppercase;
  border-radius: 4px; padding: 1.5px 7px; white-space: nowrap; }
.ri-card .pill.ok { background: #E4EAE2; color: #2D4F3A; }
.ri-card .pill.bad { background: #A33D2E; color: #FFFFFF; }
.ri-card .pill.run { background: #FFFFFF; color: #2D4F3A; border: 1px dashed #2D4F3A; }
.ri-kv { display: grid; grid-template-columns: auto 1fr; gap: 2px 12px; font-size: 12px; margin-top: 9px; }
.ri-kv .k { color: #6B6152; }
.ri-kv .v { font-family: var(--ri-mono, ui-monospace, "SF Mono", Menlo, monospace); text-align: right; color: #1A1611; }
.ri-err { margin-top: 9px; padding: 7px 9px; border-radius: 6px; background: #FFFFFF;
  border: 1px solid #DBB3A8; font: 400 11px var(--ri-mono, ui-monospace, "SF Mono", Menlo, monospace);
  color: #A33D2E; line-height: 1.45; word-break: break-word; }
</style>
"""


def _load_demo_state(ticker: str, thesis_slug: str, run_id: str) -> dict | None:
    """Find the saved demo JSON matching this run_id. The runner saves to
    `{TICKER}__{slug}__{run_id[:8]}.json`; legacy pre-suffix files are
    exactly `{TICKER}__{slug}.json`. ONLY those two names are accepted —
    a broader `{prefix}*.json` glob previously matched OTHER runs'
    suffixed files (and longer slugs sharing the prefix), silently
    rendering a different run's agent payloads in a traceability view."""
    if not DEMO_DIR.exists():
        return None
    prefix = f"{ticker.upper()}__{thesis_slug}"
    short = run_id[:8] if run_id else ""
    candidates = [DEMO_DIR / f"{prefix}__{short}.json"] if short else []
    candidates.append(DEMO_DIR / f"{prefix}.json")  # legacy exact name
    for path in candidates:
        if not path.exists():
            continue
        try:
            return json.loads(path.read_text())
        except Exception as e:
            st.warning(f"Couldn't parse {path.name}: {e}")
            return None
    return None


def _langsmith_url_for_run(run_id: str) -> str | None:
    """Build a LangSmith URL that filters the project view to this run_id.
    LangSmith indexes metadata tags, and our LangGraph runs propagate the
    `run_id` field — so a `metadata` filter on `run_id` finds the trace.

    Returns None when LANGSMITH_TRACING isn't enabled (no traces would
    exist).
    """
    if not os.environ.get("LANGSMITH_TRACING"):
        return None
    project = os.environ.get("LANGSMITH_PROJECT", "default")
    # The user's project URL pattern is opaque to us — link to the project
    # page with a free-text search for the run_id, which falls back
    # gracefully even if LangSmith's URL scheme changes. Trade-off: the
    # user lands on the search results page rather than the exact trace.
    return (
        f"https://smith.langchain.com/projects/p/{project}"
        f"?searchModel=%7B%22filter%22%3A%22{run_id}%22%7D"
    )


def _trigger_line(run_id: str) -> str:
    """Who ordered this run — the CIO decision that drilled/reused it, or
    'manual' when no cio_actions row points at it."""
    action = state_db.cio_action_for_run(run_id)
    if not action:
        return "🖱️ triggered manually (dashboard / Telegram)"
    kind = str(action.get("action") or "?")
    trigger = str(action.get("cycle_trigger") or action.get("trigger") or "?")
    cycle_ts = str(action.get("cycle_started_at") or action.get("ts") or "")[:16]
    cycle_ts = cycle_ts.replace("T", " ")
    confidence = str(action.get("confidence") or "—")
    return (
        f"🤖 triggered by CIO {md_safe(trigger)} cycle {md_safe(cycle_ts)} — "
        f"decision: <b>{md_safe(kind)}</b> (confidence {md_safe(confidence)})"
    )


def _render_run_header(run: dict) -> None:
    """Status-bordered card with the run-level summary + CIO backlink."""
    started = (run.get("started_at") or "")[:19].replace("T", " ")
    duration = run.get("duration_s") or 0.0
    status = run.get("status") or "?"
    confidence = run.get("confidence") or "—"
    ticker = run.get("ticker") or "?"
    thesis = run.get("thesis") or "?"
    run_id = run.get("run_id") or "?"
    source = run.get("source") or ""
    err = run.get("error") or ""
    err_html = (
        f"<div style='color:{FAIL}; margin-top:0.4rem;'>"
        f"⚠ {md_safe(err)}</div>"
        if err else ""
    )
    badge_color = SAGE if status in ("completed", "running") else FAIL
    st.markdown(
        f"""
        <div style="background:{EGGSHELL}; border:1px solid {TAUPE};
            border-left:6px solid {badge_color}; border-radius:6px;
            padding:1rem 1.2rem; margin-bottom:1rem;">
            <div style="color:{INK}; font-size:1.05rem;">
                <b>{html.escape(ticker)}</b> × <code>{html.escape(thesis)}</code> · {status}
                · {duration:.1f}s · confidence <b>{confidence}</b>
            </div>
            <div style="color:{INK}; opacity:0.7; font-size:0.85rem;
                margin-top:0.3rem;">
                run_id <code>{run_id}</code> · started {started} UTC
                {f"· source <code>{html.escape(source)}</code>" if source else ""}
            </div>
            <div style="color:{INK}; opacity:0.8; font-size:0.85rem;
                margin-top:0.3rem;">{_trigger_line(run_id)}</div>
            {err_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_stat_grid(
    status: str, duration_s: float, total_cost: float, total_calls: int, total_tokens: int
) -> None:
    """Five headline numbers as a bordered grid — the mockup's at-a-glance
    strip. Status colored brick when the run didn't complete cleanly."""
    status_color = SAGE if status in ("completed", "running") else FAIL
    st.markdown(
        f"""
        <div class="ri-stats">
          <div class="cell"><div class="lbl">Status</div>
            <div class="val" style="color:{status_color}">{status.upper()}</div></div>
          <div class="cell"><div class="lbl">Duration</div>
            <div class="val">{duration_s:.1f}s</div></div>
          <div class="cell"><div class="lbl">Cost</div>
            <div class="val">${total_cost:.4f}</div></div>
          <div class="cell"><div class="lbl">LLM calls</div>
            <div class="val">{total_calls}</div></div>
          <div class="cell"><div class="lbl">Tokens</div>
            <div class="val">{total_tokens:,}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_agent_cards(nodes: list[dict]) -> None:
    """Per-agent detail as a grid of cards — status pill, duration, calls,
    tokens, cost, and the error text inline for a failed agent. A node still
    `running` (schema v9 two-phase rows) gets a dashed pill."""
    if not nodes:
        st.caption("No node_runs recorded for this run.")
        return
    order = {n: i for i, n in enumerate(NODE_ORDER)}
    ordered = sorted(nodes, key=lambda n: order.get(str(n.get("node")), len(NODE_ORDER)))
    # Compact, newline-free HTML: Streamlit's markdown renderer treats any
    # indented line as a code block, so cards must be a single flat string.
    cards = []
    for n in ordered:
        status = str(n.get("status") or "?")
        failed = status == "failed"
        pill = "bad" if failed else "run" if status == "running" else "ok"
        err = str(n.get("error") or "")
        err_html = f'<div class="ri-err">{md_safe(err[:240])}</div>' if (failed and err) else ""
        cards.append(
            f'<div class="ri-card{" failed" if failed else " running" if status == "running" else ""}">'
            f'<div class="top">'
            f'<span class="nm">{md_safe(str(n.get("node") or "?"))}</span>'
            f'<span class="pill {pill}">{md_safe(status)}</span>'
            f"</div>"
            f'<div class="ri-kv">'
            f'<span class="k">duration</span><span class="v">{float(n.get("duration_s") or 0.0):.1f}s</span>'
            f'<span class="k">llm calls</span><span class="v">{int(n.get("n_calls") or 0)}</span>'
            f'<span class="k">tokens</span><span class="v">{int(n.get("tokens_in") or 0):,} / {int(n.get("tokens_out") or 0):,}</span>'
            f'<span class="k">cost</span><span class="v">${float(n.get("cost_usd") or 0.0):.4f}</span>'
            f"</div>{err_html}</div>"
        )
    st.markdown(
        f'<div class="ri-cards">{"".join(cards)}</div>', unsafe_allow_html=True
    )


def _render_failure_banner(nodes: list[dict], errors: list[dict], *, running: bool = False) -> None:
    """One glance = what broke. Names the failed agents and the first
    error so the user doesn't have to scan tables to find out."""
    failed = [n for n in nodes if n.get("status") == "failed"]
    if not failed and not errors:
        return
    first_err = ""
    if errors:
        e = errors[0]
        first_err = (
            f" First error: **{e.get('agent') or '?'}** — "
            f"{str(e.get('message') or '')[:200]}"
        )
    if failed:
        names = ", ".join(str(n.get("node") or "?") for n in failed)
        st.error(
            f"**{len(failed)} of {len(nodes)} agents failed:** {names}.{first_err}"
        )
    elif running:
        st.warning(f"Run in progress with {len(errors)} soft error(s) so far.{first_err}")
    else:
        st.warning(
            f"Run completed but logged {len(errors)} soft error(s) — "
            f"degraded output.{first_err}"
        )


def _render_timeline(nodes: list[dict]) -> None:
    """Horizontal gantt of node start→end offsets, so the fan-out /
    fan-in shape and the slow or failed agent are visible at a glance. A
    node still running is drawn up to now in taupe."""
    parsed: list[tuple[str, datetime, datetime, str]] = []
    now = datetime.now(UTC)
    for n in nodes:
        try:
            start = datetime.fromisoformat(str(n.get("started_at")))
            end = datetime.fromisoformat(str(n.get("ended_at"))) if n.get("ended_at") else now
        except (TypeError, ValueError):
            continue
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        if end.tzinfo is None:
            end = end.replace(tzinfo=UTC)
        parsed.append((str(n.get("node") or "?"), start, end, str(n.get("status") or "")))
    if not parsed:
        st.caption("No node timestamps recorded for this run.")
        return
    # Canonical graph order top-to-bottom; unknown nodes keep scan order.
    parsed.sort(
        key=lambda p: NODE_ORDER.index(p[0]) if p[0] in NODE_ORDER else len(NODE_ORDER)
    )
    t0 = min(p[1] for p in parsed)

    import matplotlib

    matplotlib.use("Agg")  # headless — GUI backends segfault off the main thread
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        figsize=(9, 0.45 * len(parsed) + 0.6), layout="constrained"
    )
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    for i, (_name, start, end, status) in enumerate(reversed(parsed)):
        left = (start - t0).total_seconds()
        width = max((end - start).total_seconds(), 0.05)
        color = SAGE if status == "completed" else TAUPE if status == "running" else FAIL
        ax.barh(i, width, left=left, height=0.55, color=color, edgecolor=TAUPE)
        label = f"{width:.1f}s" + ("  ✕" if status == "failed" else "  …" if status == "running" else "")
        ax.text(
            left + width + 0.4, i, label, va="center", fontsize=8, color=INK
        )
    ax.set_yticks(range(len(parsed)))
    ax.set_yticklabels([p[0] for p in reversed(parsed)], fontsize=9, color=INK)
    ax.set_xlabel("seconds since run start", fontsize=9, color=INK)
    ax.tick_params(colors=INK, labelsize=8)
    # Leave headroom on the right so the duration labels stay inside the axes.
    span = max((p[2] - t0).total_seconds() for p in parsed)
    ax.set_xlim(0, max(span, 1.0) * 1.18)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _render_llm_calls(snapshot: dict) -> None:
    """The local per-call trace: summary table + one expander per call with
    the FULL system prompt, user prompt and output from `llm_messages`
    (schema v9). Calls recorded before v9 fall back to their 2,000-char
    excerpts, flagged as such."""
    calls = sorted(
        (
            (str(n.get("node") or "?"), c)
            for n in snapshot.get("nodes") or []
            for c in n.get("calls") or []
        ),
        key=lambda nc: int(nc[1].get("id") or 0),
    )
    if not calls:
        st.caption(
            "No LLM calls recorded for this run — it predates schema v6 "
            "(llm_calls table), or no chat-completion call fired."
        )
        return
    rows = []
    for i, (node, c) in enumerate(calls, start=1):
        rows.append({
            "#": i,
            "node": node,
            "model": c.get("model") or "—",
            "status": c.get("status") or "completed",
            "latency_s": round(float(c.get("latency_s") or 0.0), 2),
            "tokens_in": int(c.get("tokens_in") or 0),
            "tokens_out": int(c.get("tokens_out") or 0),
            "cost_usd": round(float(c.get("cost_usd") or 0.0), 5),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
    for i, (node, c) in enumerate(calls, start=1):
        label = (
            f"call {i} — {node} · {c.get('model') or '—'} · "
            f"{float(c.get('latency_s') or 0.0):.1f}s · {c.get('status') or 'completed'}"
        )
        with st.expander(label):
            messages = c.get("messages") or []
            system = "\n\n".join(str(m.get("text") or "") for m in messages if m.get("role") == "system")
            user = "\n\n".join(
                str(m.get("text") or "")
                for m in messages
                if m.get("role") not in ("system", "assistant")
            )
            if messages:
                st.markdown(f"**System prompt** · {len(system):,} chars")
                st.code(system or "(no system message)", language=None, wrap_lines=True)
                st.markdown(f"**User prompt** · {len(user):,} chars")
                st.code(user or "(no user message)", language=None, wrap_lines=True)
            else:
                st.caption(
                    "Full messages were not stored for this call (recorded before "
                    "schema v9) — showing the clipped excerpt."
                )
                st.code(str(c.get("prompt_excerpt") or "(not captured)"), language="json", wrap_lines=True)
            if c.get("status") == "failed":
                st.error(f"Call failed: {c.get('error') or 'unknown error'}")
            response = c.get("response")
            if response is not None:
                st.markdown(f"**Output** · {len(response):,} chars")
                st.code(response, language=None, wrap_lines=True)
            elif c.get("status") == "running":
                st.caption("Output pending — the call is still in flight.")
            else:
                st.markdown("**Output (excerpt)**")
                st.code(str(c.get("response_excerpt") or "(not captured)"), language="text", wrap_lines=True)


def _render_errors(errors: list[dict]) -> None:
    if not errors:
        st.caption("No errors logged for this run.")
        return
    for e in errors:
        ts = (e.get("ts") or "")[:19].replace("T", " ")
        agent = e.get("agent") or "?"
        msg = e.get("message") or ""
        st.markdown(
            f"<div style='border-left:3px solid {TAUPE}; padding:0.4rem 0.8rem; "
            f"margin-bottom:0.4rem; background:{EGGSHELL};'>"
            f"<b>{md_safe(agent)}</b> @ {md_safe(ts)}<br>"
            f"<code style='font-size:0.85rem;'>{md_safe(msg)}</code></div>",
            unsafe_allow_html=True,
        )


def _render_agent_payload(state: dict, agent: str) -> None:
    """One agent's saved output, rendered as JSON. Wrapped in an expander
    so the page doesn't sprawl. Falls back to "(empty)" if the agent
    didn't run."""
    payload = state.get(agent)
    with st.expander(f"{agent} — saved payload"):
        if not payload:
            st.caption("(no payload — agent didn't produce output)")
            return
        if isinstance(payload, dict):
            # Show errors first if present so the user sees them at the top.
            errors = payload.get("errors") or []
            if errors:
                st.error("Errors emitted by agent:")
                for err in errors:
                    st.markdown(f"- `{err}`")
            st.json(payload, expanded=False)
        else:
            st.write(payload)


def _render_cio_action_row(a: dict) -> None:
    """One CIO action row, with an Inspect button that selects the related
    drill/reuse run in the picker above."""
    drill_id = a.get("drill_run_id") or ""
    reuse_id = a.get("reuse_run_id") or ""
    confidence = str(a.get("confidence") or "—")
    action = str(a.get("action") or "?")
    icon = {"drill": "📈", "reuse": "♻️", "dismiss": "🪦"}.get(action, "•")
    ticker = str(a.get("ticker") or "?")
    thesis = str(a.get("thesis") or "—")
    rationale = str(a.get("rationale") or "")[:240]

    line = (
        f"{icon} **{ticker}** / `{thesis}` — {action.upper()} "
        f"_(confidence: {confidence})_  \n"
        f"  {rationale}"
    )
    st.markdown(line)
    target = drill_id or reuse_id
    if target:
        kind = "drill" if drill_id else "reuse"
        if st.button(
            f"🔍 Inspect {kind} run {target[:8]}",
            key=f"cio_inspect_{a.get('id')}",
        ):
            _jump_to_run(target)


def _render_cio_cycles() -> None:
    """Render the last 10 CIO cycles with their actions inline.

    The CIO planner LLM call doesn't go through `_safe_node` (it's a
    standalone OpenRouter call inside `cio.planner.decide`), so it
    doesn't write to `node_runs`. Its telemetry footprint is the
    `cio_actions` row + the corresponding `cio_runs` rollup. This panel
    surfaces both so the user can audit the CIO's reasoning history
    next to the drill-in inspector they already use.
    """
    st.markdown("### 🤖 CIO Cycles")
    st.caption(
        "CIO heartbeat / on-demand cycles. Each cycle is one decision pass "
        "across (ticker, thesis) candidates — drills you see in the run "
        "picker above are typically *executed by* a CIO action below."
    )
    runs = state_db.recent_cio_runs(limit=10)
    if not runs:
        st.caption(
            "No CIO cycles recorded yet. Run `/cio` from Telegram or wait "
            "for the heartbeat (5am + 1pm PT)."
        )
        return

    for run in runs:
        run_id = run.get("run_id") or ""
        started = str(run.get("started_at") or "")[:19].replace("T", " ")
        trigger = str(run.get("trigger") or "")
        status = str(run.get("status") or "")
        duration = (
            f"{run['duration_s']:.1f}s" if run.get("duration_s") else "—"
        )
        n_drilled = int(run.get("n_drilled") or 0)
        n_reused = int(run.get("n_reused") or 0)
        n_dismissed = int(run.get("n_dismissed") or 0)
        title = (
            f"{trigger} · {started} · {n_drilled}d / {n_reused}r / "
            f"{n_dismissed}x · {duration} · {status}"
        )
        # Default-expand the most recent cycle so it's visible without a click.
        is_first = run_id == (runs[0].get("run_id") or "")
        with st.expander(title, expanded=is_first):
            scoped = state_db.recent_cio_actions(limit=200, cio_run_id=run_id)
            if not scoped:
                st.caption("No actions recorded for this cycle.")
                continue
            for a in scoped:
                _render_cio_action_row(a)
            if run.get("summary"):
                st.markdown("**Summary**")
                st.code(str(run["summary"])[:2000], language="text")


def _jump_to_run(run_id: str) -> None:
    """Select `run_id` in the Runs tab and switch to it on the next run. The
    tabs widget's own key cannot be written after it was instantiated in the
    same script run, so the jump is staged under `mc_jump_tab` and applied by
    `_apply_tab_jump()` before `st.tabs` renders."""
    st.session_state["inspect_run_id"] = run_id
    st.session_state["mc_jump_tab"] = "Runs"
    st.rerun()


def render_runs_tab() -> None:
    st.markdown(_RI_CSS, unsafe_allow_html=True)
    # Not cached: the picker must reflect a run that just landed.
    runs = state_db.recent_runs(limit=100)
    if not runs:
        st.warning(
            "No graph runs in `data_cache/state.db` yet. Run a drill-in from "
            "the dashboard or Telegram first."
        )
        return

    # Run picker — labels show `ticker × thesis · status · started`. Newest
    # first so the most recent run is the default.
    labels: list[str] = []
    for r in runs:
        ts = (r.get("started_at") or "")[:16].replace("T", " ")
        labels.append(
            f"{r.get('ticker', '?')} × {r.get('thesis', '?')} · "
            f"{r.get('status', '?')} · {ts} · {(r.get('run_id') or '')[:8]}"
        )

    # A stored selection may reference a run that scrolled out of the
    # 100-run window — drop it so the selectbox doesn't raise.
    if st.session_state.get("run_select") not in labels:
        st.session_state.pop("run_select", None)

    # Preselection, two sources:
    #   1. `inspect_run_id` session key — set by Health's Open buttons, the
    #      CIO-cycle Inspect buttons and the run_inspector redirect (one-shot pop).
    #   2. `?run_id=` query param — deep link from Telegram / bookmarks;
    #      only honored on first load so it can't fight a manual pick.
    preselect = st.session_state.pop("inspect_run_id", None)
    if preselect is None and "run_select" not in st.session_state:
        preselect = st.query_params.get("run_id")
    if preselect:
        matched = False
        for lbl, r in zip(labels, runs, strict=True):
            rid = str(r.get("run_id") or "")
            if rid == preselect or rid.startswith(str(preselect)):
                st.session_state["run_select"] = lbl
                matched = True
                break
        if not matched:
            st.warning(
                f"Run `{str(preselect)[:8]}…` isn't among the newest "
                f"{len(runs)} runs in the picker — showing the current "
                "selection instead."
            )

    pick_cols = st.columns([5, 2], vertical_alignment="bottom")
    chosen_label = pick_cols[0].selectbox("Run", labels, key="run_select")
    replay = pick_cols[1].toggle("▶ Replay in the control room", key="mc_replay")
    run = runs[labels.index(chosen_label)]
    run_id = run.get("run_id") or ""
    # Keep the URL shareable while this tab is on screen.
    if st.session_state.get("mc_tabs") == "Runs":
        st.query_params["run_id"] = run_id

    snapshot = state_db.run_snapshot(run_id) or {"nodes": [], "errors": []}
    nodes = snapshot.get("nodes") or []
    errors = snapshot.get("errors") or []
    total_cost = sum((n.get("cost_usd") or 0.0) for n in nodes)
    total_tokens = sum(
        (n.get("tokens_in") or 0) + (n.get("tokens_out") or 0) for n in nodes
    )
    total_calls = sum(int(n.get("n_calls") or 0) for n in nodes)

    _render_run_header(run)
    _render_stat_grid(
        str(run.get("status") or "?"),
        float(run.get("duration_s") or 0.0),
        total_cost, total_calls, total_tokens,
    )
    _render_failure_banner(nodes, errors, running=run.get("status") == "running")

    if replay:
        st.caption(
            "Replay: the same control-room renderer fed this run's stored snapshot. "
            "Drag the scrubber or press play; click a node for the prompts it saw."
        )
        render_replay(run_id)

    ls_url = _langsmith_url_for_run(run_id)
    if ls_url:
        st.link_button("🔗 Open in LangSmith", ls_url)

    # Agent timeline first — this is where the user starts when
    # something looks wrong (which node failed, which was slow, what cost).
    st.markdown("### Agent timeline")
    _render_timeline(nodes)
    _render_agent_cards(nodes)

    section_divider()

    st.markdown("### Errors logged for this run")
    _render_errors(errors)

    section_divider()

    st.markdown("### LLM call trace")
    st.caption(
        "One row per chat-completion call, recorded locally in "
        "`state.db.llm_calls`. Since schema v9 every call keeps its full "
        "system prompt, user prompt and output (`llm_messages` + "
        "`prompt_blobs`, pruned after `PROMPT_RETENTION_DAYS`); older calls "
        "show their clipped excerpts."
    )
    _render_llm_calls(snapshot)

    section_divider()

    # Agent payloads — only render if we have a saved demo state for this run.
    state = _load_demo_state(
        ticker=run.get("ticker") or "",
        thesis_slug=run.get("thesis") or "",
        run_id=run_id,
    )
    if state is None:
        st.markdown("### Agent payloads")
        st.caption(
            "No saved demo state on disk for this run. The runner writes "
            "`data_cache/demos/{TICKER}__{slug}__{run_id[:8]}.json` AFTER "
            "the graph completes; this run may have been recorded in "
            "`state.db` but not saved (older runner code, or the run "
            "failed before `_save_run_to_demo_dir` could fire)."
        )
    else:
        st.markdown("### Agent payloads (from saved state)")
        for agent in (
            "fundamentals", "filings", "news", "risk", "monte_carlo", "synthesis"
        ):
            _render_agent_payload(state, agent)

    section_divider()
    _render_cio_cycles()


# --- Health tab (pre-v2 panels, unchanged) ----------------------------------

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
    """Emoji three-way status label (used by the KPI breakdown + filter).
    Shares its classification with `_status_bits` so the two can't drift:
    'degraded' = the graph finished but an agent failed or a run-scoped
    error was logged (soft failure), previously indistinguishable from a
    clean completed run."""
    text, _, _ = _status_bits(r)
    return {
        "failed": "❌ failed",
        "degraded": "⚠️ degraded",
        "completed": "✅ completed",
    }.get(text, text)


# Graph topology order — the runs table renders one status dot per agent in
# this order so a glance shows WHERE a run broke, not just that it did.
_NODE_ORDER = NODE_ORDER
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
.mcrow .dot.run { background: #FFFFFF; border: 1.5px dashed #2D4F3A; }
.mcrow .tpill { font: 500 10.5px var(--mc-mono, ui-monospace, "SF Mono", Menlo, monospace);
  color: #6B6152; border: 1px solid #E0D5C2; border-radius: 4px; padding: 1px 6px; }
.mcrow .spill { font: 600 10px sans-serif; letter-spacing: 0.04em; text-transform: uppercase;
  border-radius: 4px; padding: 2px 8px; }
.mcrow .spill.ok { background: #E4EAE2; color: #2D4F3A; }
.mcrow .spill.deg { background: #F5EBD2; color: #9A6B1F; border: 1px solid #DCC791; }
.mcrow .spill.fail { background: #A33D2E; color: #FFFFFF; }
.mcrow .spill.run { background: #FFFFFF; color: #2D4F3A; border: 1px dashed #2D4F3A; }
</style>
"""

_RUN_COLS = [11, 1.6]  # [styled row cell, Open button] — keeps buttons aligned.


def _dots_html(status_map: dict[str, str]) -> str:
    """Colored status dots in graph order (sage ok / brick failed / dashed
    running / hollow didn't-run) — the mockup's AGENTS strip."""
    cls = {"completed": "ok", "failed": "fail", "running": "run"}
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
    if status == "running":  # schema v9: the row exists from the first node on
        return "running", "run", ""
    if status == "completed":
        if int(r.get("failed_nodes") or 0) > 0 or int(r.get("n_errors") or 0) > 0:
            return "degraded", "deg", "deg"
        return "completed", "ok", ""
    return (status or "?"), "ok", ""


def _render_runs_table(runs_all: list[dict]) -> None:
    """Filter chips + per-agent status dots + row-select → the Runs tab.
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
        # Escape DB text — the row is emitted with unsafe_allow_html, and a
        # Phase-1 ad-hoc thesis name can contain '<' (e.g. "P/E < 10").
        ticker = html.escape(str(r.get("ticker") or "?"))
        thesis = html.escape(str(r.get("thesis") or "?"))
        dur = f"{r['duration_s']:.0f}s" if r.get("duration_s") else "—"
        cost = f"${float(r.get('cost_usd') or 0.0):.4f}"
        trig = "🤖 cio" if rid in cio_run_ids else "🖱 manual"
        row_html = (
            f'<div class="mcrow {rowcls}">'
            f'<span class="mono">{started}</span>'
            f'<span><span class="tk">{ticker}</span>'
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
            _jump_to_run(rid)


def render_state_db_panel() -> None:
    """Step 5z observability — reads from data/state.py SQLite telemetry."""
    st.markdown("### Drill-in runs")
    st.caption(
        "Every full LangGraph drill-in (the dashboard's 🔍 Run drill-in "
        "button), with timing, status, cost, and per-node telemetry. Backed "
        "by `data_cache/state.db`. **Select a row to open it in the Runs "
        "tab** (agent timeline, errors, LLM call trace with full prompts)."
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


def render_health_tab() -> None:
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


# --- Page -------------------------------------------------------------------


def _apply_tab_jump() -> None:
    """Open on the Runs tab for a deep link (`?run_id=` on first load) or a
    staged in-app jump (`mc_jump_tab`, set by `_jump_to_run` and by the
    run_inspector redirect). Must run before `st.tabs` instantiates its key."""
    jump = st.session_state.pop("mc_jump_tab", None)
    if jump is None and "mc_tabs" not in st.session_state and st.query_params.get("run_id"):
        jump = "Runs"
    if jump in ("Live", "Runs", "Health"):
        st.session_state["mc_tabs"] = jump


def main() -> None:
    page_header(
        "Mission Control",
        subtitle=(
            "Live: the drill-in graph and the CIO cycle as they run, with every "
            "prompt and output. Runs: inspect or replay any past run. Health: "
            "freshness, cost, errors, evals."
        ),
    )
    _apply_tab_jump()
    tab_live, tab_runs, tab_health = st.tabs(
        ["Live", "Runs", "Health"], key="mc_tabs", on_change="rerun"
    )
    with tab_live:
        _live_fragment()
    with tab_runs:
        render_runs_tab()
    with tab_health:
        render_health_tab()


main()
