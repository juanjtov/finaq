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
  3. For each candidate: probes EDGAR's index (cheap, once per ticker per
     cycle) and auto-ingests only `auto_ingest` tickers (heartbeat →
     anchors; on-demand → the requested ticker); then calls
     `planner.decide(...)` with the probe result as `edgar_freshness` and
     a lazy news fetcher (Tavily fires only if the gates let the LLM run,
     memoised per ticker); records a `cio_actions` row. A stale ticker the
     planner picks for a drill is ingested right before that drill.
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
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cio import planner as cio_planner
from cio.planner import CIODecision, Plan
from data import state as state_db
from data import theses as theses_lifecycle
from utils import logger

if TYPE_CHECKING:  # data.freshness pulls in chromadb; keep import-time thin.
    from data.freshness import FreshnessReport

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


def _anchor_tickers() -> set[str]:
    """Union of `anchor_tickers` across curated theses — the only tickers
    the heartbeat auto-ingests (user decision 2026-09-07: a full-universe
    sweep re-embedded corpora nobody asked for, for hours)."""
    out: set[str] = set()
    for slug in _list_curated_slugs():
        thesis = _load_thesis(slug)
        if thesis:
            out.update(t.upper() for t in (thesis.get("anchor_tickers") or []))
    return out


def _fetch_news(ticker: str) -> list[dict]:
    """Best-effort recent-news pull for a single ticker. Soft-fail: any
    Tavily error returns an empty list. Caller treats `[]` as "no news,
    no signal" — the persona prompt understands this.

    Queries `"{ticker} {company name}"` with the name resolved from the
    yfinance cache (same helper the News agent uses). It used to pass the
    *thesis* name, so every ticker in the `wen` thesis searched for
    "Wendy's". Basic search depth (1 Tavily credit, not 2) and the
    planner's 14-day / 8-headline window — the planner only reads titles.

    Lazy imports so a missing TAVILY_API_KEY at start-time doesn't block
    test imports."""
    try:
        from agents.news import _company_name_for
        from data.tavily import search_news

        return search_news(
            ticker,
            _company_name_for(ticker),
            days=cio_planner.NEWS_LOOKBACK_DAYS,
            max_results=cio_planner.MAX_NEWS_HEADLINES,
            search_depth="basic",
        )
    except Exception as e:
        logger.warning(f"[cio.cio] _fetch_news({ticker}) failed: {e}")
        return []


# --- Drill execution -----------------------------------------------------


_AUTO_INGEST_TIMEOUT_S = 90.0
"""Per-ticker cap on auto-ingest inside the heartbeat. A slow EDGAR
response or chunking pass must not stall the rest of the cycle — on
timeout we log and drill on whatever's already in ChromaDB."""


def _probe_freshness(ticker: str) -> FreshnessReport | None:
    """EDGAR-index freshness probe — one ~50KB JSON round-trip, no
    download, no embedding. Runs for EVERY candidate ticker so the
    planner sees "a newer 10-Q exists at EDGAR" even when we don't
    ingest it. Soft-fail: returns None on any error."""
    from data.freshness import check_ingest_freshness

    try:
        return check_ingest_freshness(ticker)
    except Exception as e:
        logger.warning(f"[cio.cio] freshness probe raised for {ticker}: {e}")
        return None


async def _auto_ingest(ticker: str, report: FreshnessReport) -> None:
    """Download + chunk + embed the filings EDGAR has and ChromaDB lacks.

    The expensive half of freshness. The heartbeat runs it only for
    anchor tickers and for tickers the planner actually drills — a
    full sweep of the 46-ticker universe ran for hours and re-embedded
    corpora the user never asked for (2026-09-07). On-demand
    `/cio TICKER` ingests the requested ticker.

    Capped at `_AUTO_INGEST_TIMEOUT_S` to protect cycle latency.
    Soft-fails: any error / timeout logs a warning and the caller drills
    on the existing corpus. Successful auto-ingests log at INFO so
    journalctl / log aggregation can correlate them with subsequent
    cio_actions rows by timestamp.
    """
    import asyncio

    diff_str = ", ".join(
        f"{d.form}({d.chroma_date or 'missing'}→{d.edgar_date})"
        for d in report.stale_forms()
    )
    logger.info(f"[cio.cio] auto-ingest triggered for {ticker} — {diff_str}")

    try:
        from scripts.ingest_universe import ingest_ticker

        chunks = await asyncio.wait_for(
            ingest_ticker(ticker, force_refresh=True),
            timeout=_AUTO_INGEST_TIMEOUT_S,
        )
        logger.info(
            f"[cio.cio] auto-ingest for {ticker} added {chunks} chunks — "
            f"proceeding to drill"
        )
    except asyncio.TimeoutError:
        logger.warning(
            f"[cio.cio] auto-ingest for {ticker} timed out after "
            f"{_AUTO_INGEST_TIMEOUT_S}s — drilling on existing corpus"
        )
        try:
            state_db.record_error(
                agent="cio.auto_ingest",
                message=f"auto-ingest timeout for {ticker} ({diff_str})",
            )
        except Exception:
            pass
    except Exception as e:
        logger.warning(
            f"[cio.cio] auto-ingest for {ticker} failed: {e} — "
            f"drilling on existing corpus"
        )
        try:
            state_db.record_error(
                agent="cio.auto_ingest",
                message=f"auto-ingest failed for {ticker}: {e}",
            )
        except Exception:
            pass


async def _drill_one(ticker: str, thesis: dict) -> str | None:
    """Execute the existing LangGraph drill-in for a (ticker, thesis) pair.

    Returns the new graph_runs.run_id on success, None on failure. Failures
    are logged but never raised — the CIO cycle continues with the next
    pair so a single ticker outage doesn't break the heartbeat.

    Freshness / auto-ingest is NOT done here any more: `_run_cycle` probes
    every candidate before the planner decides, and `_execute_plan` runs
    `_auto_ingest` for a stale drilled ticker right before calling us.
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
    telemetry: dict | None = None,
    drill_run_id: str | None = None,
) -> None:
    """Persist a single CIODecision as a `cio_actions` row.

    `telemetry` is the dict the planner returned: model_used, tokens_in,
    tokens_out, cost_usd, latency_s. None → all zeros (e.g. for the
    drill-budget-cap demoted decisions that have no LLM call of their own).
    """
    t = telemetry or {}
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
            source=decision.source,
            model_used=t.get("model_used"),
            tokens_in=int(t.get("tokens_in") or 0),
            tokens_out=int(t.get("tokens_out") or 0),
            cost_usd=float(t.get("cost_usd") or 0.0),
            latency_s=float(t.get("latency_s") or 0.0),
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
    pair_to_telemetry: dict[tuple[str, str | None], dict],
    freshness: dict[str, FreshnessReport | None] | None = None,
    ingested: set[str] | None = None,
) -> Plan:
    """Run drills + persist every decision as a cio_actions row.

    `pair_to_thesis` maps `(ticker, thesis_slug)` → loaded thesis dict so
    we can hand the right thesis to `_drill_one` without re-loading.
    `pair_to_telemetry` maps the same key → planner-call telemetry dict
    (model_used, tokens, cost_usd, latency_s) so each cio_actions row
    captures the LLM call that produced it.

    `freshness` / `ingested` come from the cycle's probe pass: a drilled
    ticker that is stale and was not auto-ingested (non-anchor) gets
    ingested here, right before its drill — bounded by the drill budget.
    """
    freshness = freshness or {}
    ingested = ingested if ingested is not None else set()
    for d in plan.decisions:
        telemetry = pair_to_telemetry.get((d.ticker, d.thesis))
        if d.action == "drill":
            thesis = pair_to_thesis.get((d.ticker, d.thesis))
            if not thesis:
                logger.warning(
                    f"[cio.cio] missing thesis dict for ({d.ticker}, {d.thesis})"
                )
                _record_decision(
                    cio_run_id=cio_run_id, trigger=trigger, decision=d,
                    telemetry=telemetry,
                )
                continue
            report = freshness.get(d.ticker)
            if report is not None and report.is_stale and d.ticker not in ingested:
                ingested.add(d.ticker)
                await _auto_ingest(d.ticker, report)
            drill_run_id = await _drill_one(d.ticker, thesis)
            _record_decision(
                cio_run_id=cio_run_id, trigger=trigger, decision=d,
                drill_run_id=drill_run_id, telemetry=telemetry,
            )
        else:
            _record_decision(
                cio_run_id=cio_run_id, trigger=trigger, decision=d,
                telemetry=telemetry,
            )
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
    if plan.overdue_theses:
        overdue = ", ".join(
            f"{o['slug']} ({o['age_days']}d)" for o in plan.overdue_theses
        )
        lines.append(
            f"Theses overdue for review (>{theses_lifecycle.REVIEW_MAX_DAYS}d): {overdue}. "
            f"Open Theses Admin and press Mark reviewed, or edit the JSON."
        )
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
    auto_ingest: set[str] | None = None,
) -> tuple[Plan, str]:
    """Inner: open cio_run, decide each pair, cap drills, execute,
    persist, close cio_run. Returns (plan, summary_text).

    `auto_ingest`: tickers whose corpus we refresh BEFORE deciding when
    EDGAR has something newer (heartbeat → anchors; on-demand → the
    requested ticker). Every other ticker only gets the cheap EDGAR probe;
    its result reaches the planner as `edgar_freshness`, and the ingest
    happens later only if the planner picks it for a drill.
    """
    cio_run_id = state_db.start_cio_run(trigger)
    t0 = time.perf_counter()
    auto_ingest = {t.upper() for t in (auto_ingest or set())}

    pair_to_thesis: dict[tuple[str, str | None], dict] = {}
    pair_to_telemetry: dict[tuple[str, str | None], dict] = {}
    decisions: list[CIODecision] = []
    cycle_model: str | None = None

    # Per-cycle, per-ticker memos. A ticker in two theses (cross-thesis
    # multiplicity) gets one EDGAR probe and at most one Tavily call per
    # cycle, not one per pair.
    freshness: dict[str, FreshnessReport | None] = {}
    ingested: set[str] = set()
    news_cache: dict[str, list[dict]] = {}

    def _news_for(t: str) -> list[dict]:
        if t not in news_cache:
            news_cache[t] = _fetch_news(t)
        return news_cache[t]

    try:
        for ticker, slug in candidates:
            ticker = ticker.upper()
            thesis = _load_thesis(slug)
            if thesis is None:
                logger.warning(f"[cio.cio] {slug!r} not loadable — skipping {ticker}")
                continue
            pair_to_thesis[(ticker, slug)] = thesis
            # Freshness BEFORE the planner looks. The probe is cheap and
            # runs for everyone; the ingest is expensive and runs only for
            # `auto_ingest` tickers. Previously this ran inside
            # `_drill_one`, i.e. after the decision it was meant to inform.
            if ticker not in freshness:
                report = _probe_freshness(ticker)
                freshness[ticker] = report
                if report is not None and report.is_stale and ticker in auto_ingest:
                    ingested.add(ticker)
                    await _auto_ingest(ticker, report)
            try:
                decision, telemetry = cio_planner.decide(
                    ticker=ticker,
                    thesis=thesis,
                    news_fetcher=partial(_news_for, ticker),
                    edgar_freshness=freshness[ticker],
                    cooldown_hours=cooldown_hours,
                )
            except Exception as e:
                logger.error(f"[cio.cio] decide({ticker}/{slug}) failed: {e}")
                decision = CIODecision(
                    action="dismiss", ticker=ticker, thesis=slug,
                    rationale=f"planner error: {e!s}", confidence="low",
                    source="fallback",
                )
                telemetry = cio_planner._empty_telemetry()
            decisions.append(decision)
            pair_to_telemetry[(ticker.upper(), slug)] = telemetry
            if cycle_model is None and telemetry.get("model_used"):
                cycle_model = telemetry["model_used"]

        capped_decisions, n_capped = cio_planner.apply_drill_budget(
            decisions, drill_budget=drill_budget,
        )
        plan = Plan(
            decisions=capped_decisions, drill_budget=drill_budget,
            drills_capped=n_capped,
            # Nag, don't act: a thesis nobody has confirmed in 90+ days
            # is surfaced in the summary + Telegram, never auto-archived.
            overdue_theses=theses_lifecycle.overdue_theses(
                {slug for _, slug in candidates}, theses_dir=THESES_DIR,
            ),
        )

        await _execute_plan(
            cio_run_id=cio_run_id, trigger=trigger, plan=plan,
            pair_to_thesis=pair_to_thesis,
            pair_to_telemetry=pair_to_telemetry,
            freshness=freshness, ingested=ingested,
        )

        # Step 11.19 — aggregate per-cycle cost from the per-action
        # telemetry we just wrote. This is the canonical $-spent-per-CIO-cycle
        # number Mission Control surfaces; per-action rows have the breakdown.
        total_cost_usd = float(
            sum(t.get("cost_usd") or 0.0 for t in pair_to_telemetry.values())
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
            model_used=cycle_model,
            total_cost_usd=total_cost_usd,
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
    """Curated-only sweep — used by the launchd timer twice a day.
    Auto-ingests anchor tickers only; everything else is probed."""
    return await _run_cycle(
        trigger="heartbeat",
        candidates=_curated_candidates(),
        drill_budget=drill_budget,
        cooldown_hours=cooldown_hours,
        auto_ingest=_anchor_tickers(),
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
        auto_ingest=_anchor_tickers(),
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
        auto_ingest={ticker},  # the user asked about this one — refresh it
    )
