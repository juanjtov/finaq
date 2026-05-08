"""CIO planner — gates + LLM `decide()` per (ticker, thesis) pair.

Responsibilities:
  1. **Gates** — pre-LLM checks (cooldown, recent dismissal velocity, drill
     budget). The planner doesn't burn LLM tokens when a deterministic
     check already says "dismiss" / "skip the LLM".
  2. **Evidence bundle** — pulls the inputs the persona prompt expects:
     thesis JSON, RAG over past reports, recent EDGAR filings on disk,
     recent Tavily news, user notes.
  3. **LLM decide** — calls `MODEL_CIO` with the persona prompt + bundle,
     parses + Pydantic-validates the response, returns a `CIODecision`.
  4. **Plan** — orchestrates a list of pairs, applies the drill-budget cap
     (default 3), and ranks drills by confidence so the cap keeps the
     highest-conviction calls.

The planner does NOT execute drills — it only decides. `cio.cio.CIO`
(Step 11.9) takes the Plan and runs the actual graph invocations.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from cio import memory as cio_memory
from cio import rag as cio_rag
from data.chroma import _BM25_STOPWORDS
from utils import logger
from utils.models import MODEL_CIO
from utils.openrouter import get_client

# --- Constants ------------------------------------------------------------

DEFAULT_DRILL_BUDGET = 3
"""Hard cap on drills per heartbeat. Locked in v1."""

LLM_MAX_TOKENS = 700
"""Budget for one CIODecision JSON. Keep small — the schema is tiny.
Typical real response: ~200 tokens. 700 leaves headroom for verbose
rationales without inviting rambling."""

_PROMPT_PATH = Path(__file__).parent / "prompts" / "cio_persona.md"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text()

_RAG_K = 4  # Past-report sections per (ticker, thesis) for the prompt.
_NEWS_LOOKBACK_DAYS = 14
_MAX_NEWS_HEADLINES = 8  # truncate to keep prompt small.

# Watchlist-signal matching constants. The matcher is intentionally simple
# (significant-word overlap, ≥2 shared keywords for news; ≥1 for filings)
# so a domain expert reading the prompt can predict its decisions. A
# fuzzier matcher (semantic similarity, embedding distance) is in
# docs/POSTPONED.md if recall stays low after we have data.
_WATCHLIST_MIN_KEYWORD_LEN = 4
_WATCHLIST_NEWS_MIN_OVERLAP = 1
_WATCHLIST_FILING_MIN_OVERLAP = 1
_WATCHLIST_MIN_ITEM_KEYWORDS = 2  # items with <2 distinct keywords are too vague to match

# Hyphenated tokens are kept whole — `10-Q` / `10-K` / `pre-ipo` are
# meaningful as units (especially for filing-kind matching). Without
# the hyphen branch the regex would split `10-q` into "10" + "q"
# (both <4 chars, both dropped) and lose the ability to match a
# filing kind against a watchlist mention of "next 10-Q".
_WORD_RE_LOWER = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

CONFIDENCE_VALUES = ("low", "medium", "high")
ACTION_VALUES = ("drill", "reuse", "dismiss")


# --- Pydantic schemas ------------------------------------------------------


class CIODecision(BaseModel):
    """One CIO decision for a single (ticker, thesis) pair.

    `confidence` shapes downstream behaviour:
      - REUSE + high   → exec summary + "still applies as of {today}"
      - REUSE + low    → 1-line confirmation only
      - DRILL + any    → the orchestrator queues a fresh graph run
      - DISMISS + any  → no action; `rationale` lands in `cio_actions` log
    """

    action: Literal["drill", "reuse", "dismiss"]
    ticker: str
    thesis: str | None = None
    rationale: str
    reuse_run_id: str | None = None
    confidence: Literal["low", "medium", "high"] = "medium"
    followup_at: str | None = None  # ISO date the CIO suggests revisiting


class Plan(BaseModel):
    """The output of `propose_plan()`. Carries every per-pair decision +
    the drill-cap-aware rollup the orchestrator needs.

    `n_*` fields are derived from `decisions` post-cap (i.e. they reflect
    the actual execution counts, not what the LLM proposed before the
    drill-budget enforcement clipped excess drills to reuse).
    """

    decisions: list[CIODecision] = Field(default_factory=list)
    drill_budget: int = DEFAULT_DRILL_BUDGET
    drills_capped: int = 0  # how many proposed drills were demoted to reuse

    @property
    def n_drilled(self) -> int:
        return sum(1 for d in self.decisions if d.action == "drill")

    @property
    def n_reused(self) -> int:
        return sum(1 for d in self.decisions if d.action == "reuse")

    @property
    def n_dismissed(self) -> int:
        return sum(1 for d in self.decisions if d.action == "dismiss")


# --- Gate logic -----------------------------------------------------------


class GateOutcome(BaseModel):
    """Pre-LLM gate evaluation. `shortcut` is non-None when the gates can
    decide without burning tokens (e.g. 3 dismissals in 7 days → 4th is
    also dismiss). `evidence` carries the data we want the LLM to see
    *if* it gets called."""

    shortcut: CIODecision | None = None
    cooldown_status: dict
    dismissal_streak: list[dict]
    notes: str


def evaluate_gates(
    ticker: str,
    thesis: str | None,
    *,
    cooldown_hours: int = cio_memory.DEFAULT_COOLDOWN_HOURS,
) -> GateOutcome:
    """Run the deterministic gates. May short-circuit to a `dismiss`
    decision without invoking the LLM (cheap; saves tokens on yo-yo
    cases). Otherwise, return the gathered context so the LLM gets it
    in its prompt.
    """
    cooldown = cio_memory.cooldown_status(ticker, thesis, cooldown_hours=cooldown_hours)
    dismissals = cio_memory.dismissals_in_window(ticker, thesis, window_days=7)
    notes = cio_memory.thesis_notes(thesis or "")

    shortcut: CIODecision | None = None
    # Yo-yo guard: if we've dismissed this pair ≥3 times in the last 7 days,
    # short-circuit a 4th LLM call. The user can /cio TICKER force-explicit
    # to override this if they want.
    if len(dismissals) >= 3:
        shortcut = CIODecision(
            action="dismiss",
            ticker=ticker.upper(),
            thesis=thesis,
            rationale=(
                f"Yo-yo guard: {len(dismissals)} dismissals in the last 7 days for "
                f"({ticker}, {thesis}); skipping LLM until evidence shifts."
            ),
            confidence="high",
        )

    return GateOutcome(
        shortcut=shortcut,
        cooldown_status=cooldown,
        dismissal_streak=dismissals,
        notes=notes,
    )


# --- Evidence bundle (pulls past reports + news + filings) ---------------


def _summarise_news(news_items: list[dict] | None) -> list[dict]:
    """Reduce a Tavily result list to the few fields the LLM needs.

    Keeps `title`, `url`, `published_date`, optional `sentiment`. Drops
    full bodies — we just need headlines for the cadence judgement.
    """
    if not news_items:
        return []
    out: list[dict] = []
    for item in news_items[:_MAX_NEWS_HEADLINES]:
        out.append(
            {
                "title": (item.get("title") or "")[:240],
                "url": item.get("url") or "",
                "published": item.get("published_date") or "",
                "sentiment": item.get("sentiment") or "neutral",
            }
        )
    return out


def _extract_watchlist_items(text: str | None) -> list[str]:
    """Pull bullet items from a `## Watchlist` section's body.

    Synthesis writes the section as one paragraph + bullets:

        - Q3 earnings call (Aug 2026) — listen for AI capex guidance (news)
        - TSM yield disclosure in next 10-Q — supply concentration check (filings)

    We keep each bullet's full text (including the `(filings)` /
    `(news)` agent suffix), but drop the leading `- ` / `* ` / `+ `
    marker. Empty lines and non-bullet preambles are skipped.
    """
    if not text:
        return []
    items: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line[:2] in ("- ", "* ", "+ "):
            body = line[2:].strip()
            if body:
                items.append(body)
    return items


def _significant_keywords(s: str) -> set[str]:
    """Lowercase alnum tokens, length ≥ MIN, with stopwords dropped.
    Reuses the shared BM25 stopword list to stay consistent with how
    the filings RAG ranks keywords."""
    return {
        w
        for w in _WORD_RE_LOWER.findall(s.lower())
        if w not in _BM25_STOPWORDS and len(w) >= _WATCHLIST_MIN_KEYWORD_LEN
    }


def _match_watchlist_signals(
    items: list[str],
    *,
    news: list[dict] | None,
    filings: list[dict] | None,
) -> list[dict]:
    """For each watchlist item, find news headlines or filings whose
    keyword overlap meets the per-channel threshold.

    Output rows have either `matched_news_*` or `matched_filing_*` set
    so the prompt can render: "watchlist item X matched news headline Y".

    A watchlist item with fewer than `_WATCHLIST_MIN_ITEM_KEYWORDS`
    significant words is skipped — too vague to match meaningfully
    (e.g. "next quarter results"). The bar is low (4-char keyword
    minimum, 2-word overlap for news) but it's deterministic; if recall
    drops we revisit with a fuzzier matcher (see POSTPONED).
    """
    if not items:
        return []
    out: list[dict] = []
    news = news or []
    filings = filings or []
    for item in items:
        item_kws = _significant_keywords(item)
        if len(item_kws) < _WATCHLIST_MIN_ITEM_KEYWORDS:
            continue
        # News overlap.
        for n in news:
            title = n.get("title") or ""
            overlap = item_kws & _significant_keywords(title)
            if len(overlap) >= _WATCHLIST_NEWS_MIN_OVERLAP:
                out.append(
                    {
                        "watchlist_item": item,
                        "matched_news_title": title,
                        "matched_news_url": n.get("url") or "",
                        "matched_news_published": n.get("published") or "",
                        "match_keywords": sorted(overlap),
                    }
                )
        # Filing overlap — typically just kind + accession, less text. The
        # signal is "a new 10-Q landed AND watchlist mentions 10-Q-style
        # content" so we lower the threshold.
        for f in filings:
            haystack = f"{f.get('kind') or ''} {f.get('accession') or ''}"
            overlap = item_kws & _significant_keywords(haystack)
            if len(overlap) >= _WATCHLIST_FILING_MIN_OVERLAP:
                out.append(
                    {
                        "watchlist_item": item,
                        "matched_filing_kind": f.get("kind"),
                        "matched_filing_accession": f.get("accession"),
                        "matched_filing_filed_at": f.get("filed_at_iso"),
                        "match_keywords": sorted(overlap),
                    }
                )
    return out


def _summarise_recent_filings(ticker: str, since_iso: str | None) -> list[dict]:
    """Lightweight check: which 10-K / 10-Q filings landed for this
    ticker since `since_iso`? Reads file mtimes off
    `data_cache/edgar/sec-edgar-filings/{ticker}/{kind}/...` so we don't
    re-call EDGAR.

    `since_iso` is the ISO timestamp of the last drill; None means "no
    prior drill — return whatever's on disk so the LLM knows the
    corpus exists".
    """
    from data.edgar import EDGAR_DIR

    base = EDGAR_DIR / "sec-edgar-filings" / ticker.upper()
    if not base.exists():
        return []
    cutoff: datetime | None = None
    if since_iso:
        try:
            cutoff = datetime.fromisoformat(since_iso)
            if cutoff.tzinfo is None:
                cutoff = cutoff.replace(tzinfo=UTC)
        except ValueError:
            cutoff = None

    out: list[dict] = []
    for kind_dir in base.iterdir():
        if not kind_dir.is_dir():
            continue
        for accession_dir in kind_dir.iterdir():
            if not accession_dir.is_dir():
                continue
            sub = accession_dir / "full-submission.txt"
            if not sub.exists():
                continue
            mtime = datetime.fromtimestamp(sub.stat().st_mtime, tz=UTC)
            if cutoff and mtime < cutoff:
                continue
            out.append(
                {
                    "kind": kind_dir.name,
                    "accession": accession_dir.name,
                    "filed_at_iso": mtime.isoformat(),
                }
            )
    out.sort(key=lambda f: f["filed_at_iso"], reverse=True)
    return out[:6]


def build_evidence_bundle(
    *,
    ticker: str,
    thesis: dict | None,
    gates: GateOutcome,
    rag_question: str,
    news_items: list[dict] | None = None,
) -> dict:
    """Assemble the JSON context the LLM sees inside the user message.

    `thesis` is the parsed thesis JSON dict (not the slug). `news_items`
    is supplied by the caller — the planner doesn't fetch news itself
    so the orchestrator can rate-limit / cache. When `news_items=None`
    the bundle simply omits news (the LLM will see "[]" and weight
    accordingly).

    Past-report excerpts come from `cio.rag.query_past_reports` with the
    pair as a metadata pre-filter and `rag_question` as the semantic
    query (typically "thesis update / risk shift / new evidence" — the
    orchestrator picks based on the trigger).
    """
    thesis_slug = (thesis or {}).get("slug") if thesis else None
    rag_chunks = cio_rag.query_past_reports(
        question=rag_question,
        ticker=ticker,
        thesis=thesis_slug,
        k=_RAG_K,
    )
    last_report_excerpts = [
        {
            "section": str((c.get("metadata") or {}).get("section") or ""),
            "date": str((c.get("metadata") or {}).get("date") or ""),
            "run_id": str((c.get("metadata") or {}).get("run_id") or ""),
            "text": (c.get("text") or "")[:1500],
        }
        for c in rag_chunks
    ]

    last_drill_iso = (gates.cooldown_status or {}).get("last_drill_at")
    recent_filings = _summarise_recent_filings(ticker, last_drill_iso)
    recent_news = _summarise_news(news_items)

    # Watchlist-aware path. Pull the most-recent Watchlist section the
    # Synthesis agent emitted for this pair, extract bullet items, and
    # explicitly match them against the recent_news + recent_filings.
    # The matched signals are a stronger drill cue than raw news alone:
    # the prior drill already flagged this exact thing as worth tracking.
    watchlist_chunk = cio_rag.latest_watchlist_section(
        ticker=ticker, thesis=thesis_slug,
    )
    watchlist_items = _extract_watchlist_items(
        (watchlist_chunk or {}).get("text") if watchlist_chunk else None
    )
    watchlist_signals = _match_watchlist_signals(
        watchlist_items, news=recent_news, filings=recent_filings,
    )

    bundle = {
        "ticker": ticker.upper(),
        "thesis": thesis_slug,
        "thesis_summary": (thesis or {}).get("summary", "")[:500] if thesis else "",
        "material_thresholds": (thesis or {}).get("material_thresholds", []) if thesis else [],
        "cooldown_status": gates.cooldown_status,
        "recent_cio_actions": cio_memory.recent_cio_actions(ticker, thesis_slug),
        "last_report_excerpts": last_report_excerpts,
        "watchlist_items": watchlist_items,
        "watchlist_signals": watchlist_signals,
        "recent_filings": recent_filings,
        "recent_news": recent_news,
        "notes": gates.notes,
    }
    return bundle


# --- LLM decide -----------------------------------------------------------


def _extract_json_object(raw: str) -> str | None:
    """Best-effort grab of the first balanced `{...}` in raw text.

    The persona prompt says "JSON only" but LLMs occasionally wrap with
    ```json fences or sneak a preamble. We retry the parse on the
    extracted blob if strict json.loads fails on the full string.
    """
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    return m.group(0) if m else None


def _parse_decision(raw: str) -> tuple[CIODecision | None, str | None]:
    """Parse + validate the LLM response. Returns (decision, error_msg).

    On any parse / validation failure we surface the error string so the
    caller can record it on the `cio_actions` row's rationale and
    fall back to a deterministic dismiss.
    """
    if not raw or not raw.strip():
        return None, "empty LLM response"
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        snippet = _extract_json_object(raw)
        if not snippet:
            return None, "no JSON object found in LLM response"
        try:
            obj = json.loads(snippet)
        except json.JSONDecodeError as e:
            return None, f"json parse failed: {e}"
    try:
        return CIODecision.model_validate(obj), None
    except ValidationError as e:
        return None, f"schema validation failed: {e}"


def _call_llm(*, system_prompt: str, evidence: dict) -> str:
    """Single OpenRouter chat-completion call. Returns raw text.

    The interceptor in `utils.openrouter` populates the per-node
    telemetry ContextVar so this call shows up in the Run Inspector
    when invoked from inside `_safe_node` (Step 11.15 wires that up).
    """
    client = get_client()
    user_payload = json.dumps(evidence, indent=2, default=str)
    resp = client.chat.completions.create(
        model=MODEL_CIO,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Decide for this pair. Output ONLY the JSON object — no "
                    "fences, no commentary.\n\n"
                    f"```json\n{user_payload}\n```"
                ),
            },
        ],
        max_tokens=LLM_MAX_TOKENS,
        temperature=0.0,
    )
    if not resp.choices:
        return ""
    return resp.choices[0].message.content or ""


def decide(
    *,
    ticker: str,
    thesis: dict | None,
    rag_question: str = "What has changed since the last drill-in?",
    news_items: list[dict] | None = None,
    cooldown_hours: int = cio_memory.DEFAULT_COOLDOWN_HOURS,
    system_prompt: str | None = None,
) -> CIODecision:
    """End-to-end per-pair decision: gate → bundle → LLM → parse.

    Always returns a `CIODecision`: a parse failure becomes a deterministic
    `dismiss` with the error in the rationale (so the run never hangs on
    a flaky LLM response).

    `system_prompt` defaults to the persona file. Tests pass a stub.
    """
    thesis_slug = (thesis or {}).get("slug") if isinstance(thesis, dict) else None
    gates = evaluate_gates(ticker, thesis_slug, cooldown_hours=cooldown_hours)
    if gates.shortcut is not None:
        return gates.shortcut

    bundle = build_evidence_bundle(
        ticker=ticker, thesis=thesis, gates=gates,
        rag_question=rag_question, news_items=news_items,
    )

    try:
        raw = _call_llm(
            system_prompt=system_prompt or _SYSTEM_PROMPT,
            evidence=bundle,
        )
    except Exception as e:
        logger.error(f"[cio.planner] LLM call failed for {ticker}/{thesis_slug}: {e}")
        return CIODecision(
            action="dismiss",
            ticker=ticker.upper(),
            thesis=thesis_slug,
            rationale=f"LLM call failed: {e!s}",
            confidence="low",
        )

    decision, err = _parse_decision(raw)
    if decision is None:
        logger.warning(f"[cio.planner] parse failed for {ticker}/{thesis_slug}: {err}")
        return CIODecision(
            action="dismiss",
            ticker=ticker.upper(),
            thesis=thesis_slug,
            rationale=f"LLM response unparseable: {err}",
            confidence="low",
        )

    # Force ticker / thesis to match what we asked about — the LLM has a
    # bad habit of retyping inputs and occasionally normalises ticker
    # case. The orchestrator uses `decision.ticker` to route, so be
    # strict here.
    decision.ticker = ticker.upper()
    if thesis_slug and not decision.thesis:
        decision.thesis = thesis_slug

    # If REUSE was chosen but no run_id was supplied, fall back to the
    # latest drill we know about. Otherwise the orchestrator would have
    # nothing to reuse.
    if decision.action == "reuse" and not decision.reuse_run_id:
        decision.reuse_run_id = (gates.cooldown_status or {}).get("last_drill_run_id")

    return decision


# --- Plan-level: cap drills + rank --------------------------------------


def apply_drill_budget(
    decisions: list[CIODecision],
    drill_budget: int = DEFAULT_DRILL_BUDGET,
) -> tuple[list[CIODecision], int]:
    """Cap drills at `drill_budget`; demote the lowest-confidence excess
    drills to `reuse` when a recent report exists, else `dismiss`.

    Returns `(updated_decisions, n_capped)`. `n_capped` is the number of
    drills that were demoted — surfaces in the exec summary so the user
    knows budget was a binding constraint.
    """
    if drill_budget < 0:
        raise ValueError(f"drill_budget must be >= 0, got {drill_budget!r}")

    drills = [d for d in decisions if d.action == "drill"]
    others = [d for d in decisions if d.action != "drill"]

    if len(drills) <= drill_budget:
        return decisions, 0

    # Rank drills high → medium → low. Stable on ties (preserves caller order).
    confidence_order = {"high": 0, "medium": 1, "low": 2}
    drills_sorted = sorted(drills, key=lambda d: confidence_order.get(d.confidence, 3))

    kept = drills_sorted[:drill_budget]
    excess = drills_sorted[drill_budget:]

    demoted: list[CIODecision] = []
    for d in excess:
        # Try to reuse the most recent drill for this pair if we have one;
        # otherwise dismiss. Either way we record the budget reason in the
        # rationale so the audit log is honest.
        cooldown = cio_memory.cooldown_status(d.ticker, d.thesis)
        last_run = (cooldown or {}).get("last_drill_run_id")
        new_action = "reuse" if last_run else "dismiss"
        demoted.append(
            CIODecision(
                action=new_action,
                ticker=d.ticker,
                thesis=d.thesis,
                rationale=(
                    f"[budget cap] originally `drill` ({d.confidence}); demoted to "
                    f"`{new_action}` because heartbeat budget = {drill_budget}. "
                    f"Original rationale: {d.rationale}"
                ),
                reuse_run_id=last_run if new_action == "reuse" else None,
                confidence="low",
                followup_at=d.followup_at,
            )
        )

    # Preserve original input ordering.
    new_decisions: list[CIODecision] = []
    kept_ids = {(d.ticker, d.thesis) for d in kept}
    demoted_map = {(d.ticker, d.thesis): d for d in demoted}
    for d in decisions:
        if d.action != "drill":
            new_decisions.append(d)
        elif (d.ticker, d.thesis) in kept_ids:
            new_decisions.append(d)
        else:
            new_decisions.append(demoted_map[(d.ticker, d.thesis)])

    return new_decisions, len(demoted)
