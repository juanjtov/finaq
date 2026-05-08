"""CIO orchestrator — the persona-driven decide+execute layer.

Sits above the existing LangGraph pipeline. The graph stays unchanged; the
CIO calls `agents.invoke_with_telemetry(graph, state)` as a tool when (and
only when) its planner picks `action=drill` for a (ticker, thesis) pair.

Public entry points (used by `cio.dispatcher` + the Telegram `/cio` handler):

  - `run_heartbeat()`           — sweep every curated thesis ticker
  - `run_on_demand(ticker, thesis_slug=None)`  — single-ticker /cio TICKER
  - `run_catchup()`             — same as heartbeat, but tagged `trigger='catchup'`
                                   so dashboards surface "this was a catch-up cycle"

Each entry point:

  1. Opens a `cio_runs` row (telemetry parent).
  2. Builds a candidate list (ticker, thesis_slug) per `_curated_candidates()`.
  3. For each candidate: pulls news (Tavily, soft-fail), calls
     `planner.decide(...)`, records a `cio_actions` row.
  4. Applies the drill-budget cap (post-LLM).
  5. Executes drills via the existing graph; records reuse / dismiss
     directly without further computation.
  6. Composes an exec summary, fires Telegram + Notion sends (Step 11.10
     wires `cio.notify`; until then we just log + persist the summary).
  7. Closes the `cio_runs` row with rolled-up counts + summary.

Constraints we enforce here (not in the planner):
  - **Drill budget**: hard cap (default 3) applies only to heartbeat /
    catchup. On-demand `/cio TICKER` is exempt — the user explicitly asked.
  - **Curated-only sweep**: heartbeat skips `adhoc_*` theses by filename
    prefix. On-demand can target any thesis (curated or adhoc).
  - **Telemetry**: every LLM call inside `_safe_node` is captured by the
    ContextVar accumulator. The CIO planner LLM call goes through that
    same path so the Run Inspector (Step 11.15) can show CIO calls.

Tests stub:
  - `planner.decide`            — returns canned CIODecisions
  - `_drill_one`                — to avoid running the real graph
  - `_fetch_news`               — to avoid Tavily calls
  - `cio_notify.send_summary`   — to avoid Telegram / Notion
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cio import planner as cio_planner
from cio.planner import CIODecision, Plan
from data import state as state_db
from data import theses as theses_lifecycle
from utils import logger

THESES_DIR = Path("theses")
ADHOC_PREFIX = theses_lifecycle.ADHOC_PREFIX

# Curated theses we sweep on heartbeat. Anything not in this set still
# works on-demand via /cio TICKER thesis.
def _list_curated_slugs() -> list[str]:
    """Slugs of every curated (non-adhoc) thesis JSON in /theses/."""
    if not THESES_DIR.exists():
        return []
    return sorted(
        p.stem
        for p in THESES_DIR.glob("*.json")
        if not p.stem.startswith(ADHOC_PREFIX)
    )


def _load_thesis(slug: str) -> dict | None:
    """Read + parse `theses/{slug}.json`. Returns None on error.

    Adds `slug` to the dict so downstream consumers (planner.decide,
    summary builder) can identify the thesis without an extra arg.
    """
    path = THESES_DIR / f"{slug}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        data["slug"] = slug
        return data
    except Exception as e:
        logger.warning(f"[cio.cio] _load_thesis({slug!r}) failed: {e}")
        return None


def _curated_candidates() -> list[tuple[str, str]]:
    """All (ticker, thesis_slug) pairs the heartbeat sweeps.

    Iterates every curated thesis, takes its `universe` tickers, and emits
    a (ticker, slug) tuple per pair. Drops duplicates within the SAME
    thesis but keeps cross-thesis multiplicity (e.g. NVDA appears in both
    `ai_cake` and `nvda_halo`) — the planner gets both perspectives.
    """
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for slug in _list_curated_slugs():
        thesis = _load_thesis(slug)
        if not thesis:
            continue
        # Heartbeat sweeps anchor tickers first (highest priority), then
        # the remainder of the universe. The planner doesn't see ordering
        # so it doesn't matter beyond budget-cap behavior — but ordering
        # by anchor first means the drill budget tends to land on
        # anchors when multiple decisions tie on confidence.
        anchors = [t.upper() for t in (thesis.get("anchor_tickers") or [])]
        rest = [t.upper() for t in (thesis.get("universe") or []) if t.upper() not in set(anchors)]
        for ticker in anchors + rest:
            pair = (ticker, slug)
            if pair in seen:
                continue
            seen.add(pair)
            out.append(pair)
    return out


def _fetch_news(ticker: str, company_name: str | None = None) -> list[dict]:
    """Best-effort recent-news pull for a single ticker. Soft-fail: any
    Tavily error returns an empty list. Caller treats `[]` as "no news,
    no signal" — the persona prompt understands this.

    Lazy import so a missing TAVILY_API_KEY at start-time doesn't block
    test imports."""
    try:
        from data.tavily import search_news

        return search_news(ticker, company_name)
    except Exception as e:
        logger.warning(f"[cio.cio] _fetch_news({ticker}) failed: {e}")
        return []


# --- Drill execution -----------------------------------------------------


async def _drill_one(ticker: str, thesis: dict) -> str | None:
    """Execute the existing LangGraph drill-in for a (ticker, thesis) pair.

    Returns the new graph_runs.run_id on success, None on failure. Failures
    are logged but never raised — the CIO cycle continues with the next
    pair so a single ticker outage doesn't break the heartbeat.
    """
    try:
        # Lazy: graph build pulls in every agent module, including the
        # heavy LLM clients. Defer until we actually need to drill.
        from agents import build_graph, invoke_with_telemetry

        graph = build_graph()
        final = await invoke_with_telemetry(
            graph, {"ticker": ticker.upper(), "thesis": thesis}
        )
        return final.get("run_id")
    except Exception as e:
        logger.error(f"[cio.cio] _drill_one({ticker}, {thesis.get('slug')}) failed: {e}")
        return None


# --- Cycle orchestration -------------------------------------------------


def _record_decision(
    *,
    cio_run_id: str,
    trigger: str,
    decision: CIODecision,
    drill_run_id: str | None = None,
) -> None:
    """Persist a single CIODecision as a `cio_actions` row."""
    try:
        state_db.record_cio_action(
            cio_run_id=cio_run_id,
            trigger=trigger,
            ticker=decision.ticker,
            thesis=decision.thesis,
            action=decision.action,
            rationale=decision.rationale,
            drill_run_id=drill_run_id,
            reuse_run_id=decision.reuse_run_id if decision.action == "reuse" else None,
            confidence=decision.confidence,
            decision_json=decision.model_dump_json(),
        )
    except Exception as e:
        logger.warning(
            f"[cio.cio] record_cio_action failed for "
            f"{decision.ticker}/{decision.thesis}: {e}"
        )


async def _execute_plan(
    *,
    cio_run_id: str,
    trigger: str,
    plan: Plan,
    pair_to_thesis: dict[tuple[str, str | None], dict],
) -> Plan:
    """Run drills + persist every decision as a cio_actions row.

    `pair_to_thesis` maps `(ticker, thesis_slug)` → loaded thesis dict so
    we can hand the right thesis to `_drill_one` without re-loading.
    """
    for d in plan.decisions:
        if d.action == "drill":
            thesis = pair_to_thesis.get((d.ticker, d.thesis))
            if not thesis:
                logger.warning(
                    f"[cio.cio] missing thesis dict for ({d.ticker}, {d.thesis})"
                )
                _record_decision(
                    cio_run_id=cio_run_id, trigger=trigger, decision=d,
                )
                continue
            drill_run_id = await _drill_one(d.ticker, thesis)
            _record_decision(
                cio_run_id=cio_run_id, trigger=trigger, decision=d,
                drill_run_id=drill_run_id,
            )
        else:
            _record_decision(cio_run_id=cio_run_id, trigger=trigger, decision=d)
    return plan


def _compose_summary(plan: Plan, *, trigger: str, duration_s: float) -> str:
    """Plain-text exec summary stored on the cio_runs row.

    Step 11.10's `cio.notify` will format this richer for Telegram /
    Notion; this is the canonical text the dashboards surface.
    """
    lines: list[str] = []
    lines.append(
        f"CIO {trigger} cycle — {plan.n_drilled} drilled, "
        f"{plan.n_reused} reused, {plan.n_dismissed} dismissed "
        f"(budget cap demoted {plan.drills_capped})."
    )
    lines.append(f"Duration: {duration_s:.1f}s.")
    if plan.decisions:
        lines.append("")
        lines.append("Decisions:")
    for d in plan.decisions:
        thesis_part = f" / {d.thesis}" if d.thesis else ""
        lines.append(
            f"  • {d.action.upper()}: {d.ticker}{thesis_part} "
            f"({d.confidence}) — {d.rationale[:200]}"
        )
    return "\n".join(lines)


async def _run_cycle(
    *,
    trigger: str,
    candidates: list[tuple[str, str]],
    drill_budget: int,
    cooldown_hours: int = 48,
) -> tuple[Plan, str]:
    """Inner: open cio_run, decide each pair, cap drills, execute,
    persist, close cio_run. Returns (plan, summary_text).
    """
    cio_run_id = state_db.start_cio_run(trigger)
    t0 = time.perf_counter()

    pair_to_thesis: dict[tuple[str, str | None], dict] = {}
    decisions: list[CIODecision] = []

    try:
        for ticker, slug in candidates:
            thesis = _load_thesis(slug)
            if thesis is None:
                logger.warning(f"[cio.cio] {slug!r} not loadable — skipping {ticker}")
                continue
            pair_to_thesis[(ticker.upper(), slug)] = thesis
            news = _fetch_news(ticker, thesis.get("name"))
            try:
                decision = cio_planner.decide(
                    ticker=ticker,
                    thesis=thesis,
                    news_items=news,
                    cooldown_hours=cooldown_hours,
                )
            except Exception as e:
                logger.error(f"[cio.cio] decide({ticker}/{slug}) failed: {e}")
                decision = CIODecision(
                    action="dismiss", ticker=ticker.upper(), thesis=slug,
                    rationale=f"planner error: {e!s}", confidence="low",
                )
            decisions.append(decision)

        capped_decisions, n_capped = cio_planner.apply_drill_budget(
            decisions, drill_budget=drill_budget,
        )
        plan = Plan(
            decisions=capped_decisions, drill_budget=drill_budget,
            drills_capped=n_capped,
        )

        await _execute_plan(
            cio_run_id=cio_run_id, trigger=trigger, plan=plan,
            pair_to_thesis=pair_to_thesis,
        )

        duration_s = time.perf_counter() - t0
        summary = _compose_summary(plan, trigger=trigger, duration_s=duration_s)
        state_db.finish_cio_run(
            cio_run_id, "completed",
            duration_s=duration_s,
            n_actions=len(plan.decisions),
            n_drilled=plan.n_drilled,
            n_reused=plan.n_reused,
            n_dismissed=plan.n_dismissed,
            summary=summary,
        )
        return plan, summary
    except Exception as e:
        duration_s = time.perf_counter() - t0
        logger.error(f"[cio.cio] cycle ({trigger}) failed: {e}")
        state_db.finish_cio_run(
            cio_run_id, "failed", error=str(e), duration_s=duration_s,
        )
        raise


# --- Public entry points --------------------------------------------------


async def run_heartbeat(
    *,
    drill_budget: int = cio_planner.DEFAULT_DRILL_BUDGET,
    cooldown_hours: int = 48,
) -> tuple[Plan, str]:
    """Curated-only sweep — used by the launchd timer twice a day."""
    return await _run_cycle(
        trigger="heartbeat",
        candidates=_curated_candidates(),
        drill_budget=drill_budget,
        cooldown_hours=cooldown_hours,
    )


async def run_catchup(
    *,
    drill_budget: int = cio_planner.DEFAULT_DRILL_BUDGET,
    cooldown_hours: int = 48,
) -> tuple[Plan, str]:
    """Same as heartbeat, tagged `catchup` — fires at boot when
    `last_successful_cio_run_at()` is >8h old."""
    return await _run_cycle(
        trigger="catchup",
        candidates=_curated_candidates(),
        drill_budget=drill_budget,
        cooldown_hours=cooldown_hours,
    )


async def run_on_demand(
    ticker: str,
    thesis_slug: str | None = None,
    *,
    drill_budget: int | None = None,
    cooldown_hours: int = 48,
) -> tuple[Plan, str]:
    """`/cio TICKER` or `/cio TICKER thesis` — single-pair (or single-ticker)
    plan. Drill budget is unlimited by default (the user explicitly asked).

    Behaviour:
      - thesis_slug given: decide for exactly (ticker, thesis_slug).
      - thesis_slug None: find every thesis whose universe contains ticker
        (curated OR adhoc) and decide for each. Each gets its own action
        recorded.

    The user can pass an `adhoc_*` slug too — on-demand isn't
    curated-only, only the heartbeat is.
    """
    ticker = ticker.upper()
    candidates: list[tuple[str, str]] = []
    if thesis_slug:
        candidates = [(ticker, thesis_slug)]
    else:
        # Search every thesis (curated + adhoc) for ticker membership.
        for slug in sorted(p.stem for p in THESES_DIR.glob("*.json")):
            t = _load_thesis(slug)
            if not t:
                continue
            universe = {x.upper() for x in (t.get("universe") or [])}
            if ticker in universe:
                candidates.append((ticker, slug))
        if not candidates:
            # Fall back to the `general` thesis if it exists, so we never
            # silently drop an on-demand request.
            if (THESES_DIR / "general.json").exists():
                candidates = [(ticker, "general")]

    budget = drill_budget if drill_budget is not None else len(candidates)
    return await _run_cycle(
        trigger="on_demand",
        candidates=candidates,
        drill_budget=budget,
        cooldown_hours=cooldown_hours,
    )
