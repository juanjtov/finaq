"""Discovery agent (Phase 2) — propose-then-verify halo-graph builder.

See ARCHITECTURE §13. Two stages:

  (A) Propose  — one LLM call (model resolved via `MODEL_DISCOVERY`) turns a
      free-text TOPIC or a seed TICKER into a candidate `Thesis`: a universe of
      public tickers plus a *generously* proposed set of relationships. Reuses
      the ad-hoc-thesis JSON-parsing idioms.
  (B) Ground   — each proposed relationship is checked against the two
      companies' real SEC filings (`data.vectors`) and recent news
      (`data.finnhub`). Corroborated edges are kept with a confidence score and
      citations; the rest are pruned from the emitted thesis. B grounds A.

Output: a grounded `Thesis` saved to `theses/adhoc_{slug}.json` (kept off the
CIO heartbeat until promoted — §13.4), plus the full proposed+grounded graph
persisted to `state.db` via `data.graph`.

This is a library module, not a drill-in graph node (§13.5b): the public entry
point is `async def discover(topic=..., ticker=...) -> DiscoveryResult`. Its
single proposal LLM call therefore isn't captured in `node_runs`, the same
accepted trade-off as `agents.adhoc_thesis` and the CIO planner.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from agents.adhoc_thesis import adhoc_slug
from agents.news import _company_name_for
from data import finnhub, graph, vectors
from utils import logger
from utils.models import MODEL_DISCOVERY
from utils.openrouter import get_client
from utils.schemas import Evidence, GraphEdge, GraphNode, Thesis

THESES_DIR = Path("theses")

_PROMPT_PATH = Path(__file__).parent / "prompts" / "discovery.md"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text()

LLM_MAX_TOKENS = 6000
"""Matches the ad-hoc synthesizer's budget — worst case is a 15-ticker
universe with 15 relationships and 10 thresholds."""

# --- Grounding knobs (ARCHITECTURE §13.2) ---------------------------------
GROUND_THRESHOLD = 0.3
"""Minimum confidence for an edge to be kept in the emitted thesis. A single
first-party filing co-mention (FILING_WEIGHT) clears it comfortably; a lone
news co-mention (NEWS_WEIGHT) just reaches it. Zero-corroboration edges are
pruned from the thesis but still stored in the graph with grounded=0."""
FILING_WEIGHT = 0.5  # per direction: `from`'s filing naming `to`, and vice-versa
NEWS_WEIGHT = 0.3  # co-mention in `from`'s recent news
NEWS_DAYS = 90
NEWS_MAX_RESULTS = 10
FILINGS_K = 5

# Hard cap on universe size. The prompt asks for 6-8; this is the guarantee
# regardless of what the model returns. Whether a larger universe yields better
# discovery is an open question — tracked as an eval-suite item in POSTPONED.
MAX_UNIVERSE = 8

# A plausible US ticker: leading letter, then up to 6 of letter/digit/./-.
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,6}$")
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
# Words that never distinguish a company for co-mention matching.
_NAME_STOP = {"THE", "AND"}
# Real tickers that are also common English words: matched as bare symbols in
# prose they produce false co-mentions, so for these we require the company-name
# token instead of the bare ticker. A denylist heuristic (not exhaustive) — the
# durable fix is the deferred per-edge LLM adjudication (ARCHITECTURE §13.2).
# 1-2 char tickers (A, ON, IT, GO, ...) are handled by the length>=3 guard in
# `_mention_hit`, so they don't need listing here.
_TICKER_STOPWORDS = {
    "ALL",
    "AND",
    "ANY",
    "ARE",
    "BIG",
    "CAR",
    "CAT",
    "DAY",
    "FOR",
    "GET",
    "GOOD",
    "HAS",
    "HOW",
    "KEY",
    "LOW",
    "MAN",
    "NEW",
    "NOW",
    "ONE",
    "OUT",
    "OWN",
    "PLAN",
    "REAL",
    "RUN",
    "SEE",
    "THE",
    "USE",
    "WHO",
    "WHY",
}


@dataclass
class DiscoveryResult:
    """Return value from `discover`. Carries the grounded thesis plus the
    graph-build stats a CLI / UI wants to show."""

    slug: str
    thesis: Thesis | None
    path: Path
    n_universe: int = 0
    n_edges_proposed: int = 0
    n_edges_grounded: int = 0
    dropped_tickers: list[str] = field(default_factory=list)
    cached: bool = False
    error: str | None = None


# --- LLM proposal (stage A) -----------------------------------------------


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        nl = text.find("\n")
        if nl > 0:
            text = text[nl + 1 :]
        if text.endswith("```"):
            text = text[:-3].rstrip()
    return text.strip()


def _parse_response(raw: str) -> dict:
    """Parse the proposal into a dict — strict `json.loads`, then a regex
    fallback for when the model wraps the JSON in prose. `{}` on failure."""
    cleaned = _strip_code_fences(raw)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    m = _JSON_OBJECT_RE.search(cleaned)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            logger.warning(
                f"[discovery] regex-extracted JSON failed to parse; "
                f"raw len={len(raw)}, first 200: {raw[:200]!r}"
            )
    else:
        logger.warning(
            f"[discovery] no JSON object in response; raw len={len(raw)}, "
            f"first 300: {raw[:300]!r}"
        )
    return {}


def _build_user_prompt(*, topic: str | None, ticker: str | None) -> str:
    if ticker:
        return (
            f"Mode: TICKER\nInput: {ticker.upper()}\n\n"
            f"Build a halo graph around {ticker.upper()}. It MUST be the first "
            f"anchor and present in the universe. Propose its suppliers, "
            f"customers, peers, and competitors, and draw edges radiating from it."
        )
    return (
        f"Mode: TOPIC\nInput: {topic}\n\n"
        f"Build a halo graph for this topic: 6-15 representative public tickers "
        f"spanning the value chain, 1-3 pure-play anchors, and a generous set of "
        f"candidate relationships (they will be verified against filings + news)."
    )


def _propose(*, topic: str | None, ticker: str | None) -> tuple[dict, str]:
    """Single proposal LLM call. Returns `(parsed_dict, raw_response)`.
    Synchronous — the async entry point offloads it via `asyncio.to_thread`."""
    client = get_client()
    resp = client.chat.completions.create(
        model=MODEL_DISCOVERY,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(topic=topic, ticker=ticker)},
        ],
        max_tokens=LLM_MAX_TOKENS,
    )
    raw = (resp.choices[0].message.content or "").strip()
    logger.info(f"[discovery] proposal received: {len(raw)} chars")
    return _parse_response(raw), raw


# --- Grounding (stage B) --------------------------------------------------


def _distinctive_name_token(name: str) -> str:
    """First distinctive word of a company name (skip 'The'/'And', require
    length >= 3), used for a loose co-mention match. '' if none."""
    for word in re.split(r"[^A-Za-z]+", name or ""):
        wu = word.upper()
        if len(wu) >= 3 and wu not in _NAME_STOP:
            return wu
    return ""


def _mention_hit(text: str, ticker: str, name: str) -> bool:
    """True when `text` co-mentions the company: its distinctive company-name
    token, or its ticker symbol. The bare-ticker match is used only for symbols
    unambiguous in prose (>= 3 chars and not a common English word), so a ticker
    like `A` / `ON` / `IT` / `ALL` / `CAT` can't be "grounded" by ordinary text
    — for those only the name token counts. Confirming an edge's direction/type
    (beyond mere co-occurrence) is the deferred LLM-adjudication step (§13.2)."""
    hay = (text or "").upper()
    token = _distinctive_name_token(name)
    if token and re.search(rf"\b{re.escape(token)}\b", hay):
        return True
    tk = ticker.upper()
    if len(tk) >= 3 and tk not in _TICKER_STOPWORDS:
        return bool(re.search(rf"\b{re.escape(tk)}\b", hay))
    return False


def _name_map(tickers: list[str]) -> dict[str, str]:
    """Resolve each ticker to a company name via the yfinance cache (§6.7).
    Falls back to the ticker itself when unresolved."""
    return {t: _company_name_for(t) for t in tickers}


def _filings_corroboration(a: str, b: str, b_name: str, edge_type: str) -> Evidence | None:
    """Does `a`'s filing corpus name `b`? Best-effort — only when `a` is
    ingested. Returns one Evidence citation on a hit, else None."""
    try:
        if not vectors.has_ticker(a):
            return None
        question = f"{b_name} {b} {edge_type} relationship supplier customer partner"
        chunks = vectors.query(a, question, k=FILINGS_K)
    except Exception as e:  # noqa: BLE001 — grounding is best-effort, never fatal
        logger.info(f"[discovery] filings grounding skipped for {a}->{b}: {e}")
        return None
    for ch in chunks:
        text = ch.get("text", "")
        if _mention_hit(text, b, b_name):
            md = ch.get("metadata") or {}
            return Evidence(
                source="edgar",
                accession=md.get("accession"),
                item=md.get("item_code") or md.get("item"),
                excerpt=text[:300],
                as_of=md.get("filed_date"),
                note=f"{a} filing co-mentions {b}",
            )
    return None


def _news_corroboration(
    frm: str, to: str, to_name: str, news_cache: dict[str, list]
) -> Evidence | None:
    """Does any recent article about `frm` co-mention `to`? Cached per `frm`
    within a run. Returns one Evidence citation on a hit, else None."""
    articles = news_cache.get(frm)
    if articles is None:
        try:
            articles = finnhub.search_news(
                frm,
                None,
                days=NEWS_DAYS,
                max_results=NEWS_MAX_RESULTS,
                with_body=False,
            )
        except Exception as e:  # noqa: BLE001 — best-effort
            logger.info(f"[discovery] news grounding failed for {frm}: {e}")
            articles = []
        news_cache[frm] = articles
    for art in articles:
        blob = f"{art.get('title', '')} {art.get('content', '')}"
        if _mention_hit(blob, to, to_name):
            return Evidence(
                source=art.get("source") or "news",
                url=art.get("url"),
                excerpt=(art.get("title") or "")[:200],
                as_of=art.get("published_date"),
                note=f"{frm} news co-mentions {to}",
            )
    return None


def _ground_edge(
    frm: str,
    to: str,
    edge_type: str,
    name_map: dict[str, str],
    news_cache: dict[str, list],
) -> tuple[float, list[Evidence]]:
    """Score one edge in [0, 1] and collect its supporting citations. Filings
    are checked both directions; news is checked from the `from` side."""
    to_name = name_map.get(to, to)
    frm_name = name_map.get(frm, frm)
    confidence = 0.0
    evidence: list[Evidence] = []
    for a, b, b_name in ((frm, to, to_name), (to, frm, frm_name)):
        ev = _filings_corroboration(a, b, b_name, edge_type)
        if ev is not None:
            confidence += FILING_WEIGHT
            evidence.append(ev)
    news_ev = _news_corroboration(frm, to, to_name, news_cache)
    if news_ev is not None:
        confidence += NEWS_WEIGHT
        evidence.append(news_ev)
    return min(confidence, 1.0), evidence


# --- Filings auto-ingest (opt-in) -----------------------------------------


async def _ensure_ingested(tickers: list[str]) -> None:
    """Ingest each ticker's filings that aren't already in the manifest, so
    filings-grounding has a corpus to check. Sequential + best-effort (EDGAR is
    rate-limited); a ticker whose ingest fails just stays news-only.

    `ingest_ticker` is imported lazily to keep `agents/` decoupled from
    `scripts/` (and to avoid pulling the ingest deps unless the flag is used)."""
    from scripts.ingest_universe import ingest_ticker

    for t in tickers:
        try:
            if vectors.has_ticker(t):
                continue
            logger.info(f"[discovery] auto-ingesting filings for {t}")
            n = await ingest_ticker(t)
            logger.info(f"[discovery] {t}: ingested {n} chunks")
        except Exception as e:  # noqa: BLE001 — best-effort; ticker stays news-only
            logger.warning(f"[discovery] auto-ingest failed for {t}: {e}")


# --- Ticker hygiene -------------------------------------------------------


def _clean_tickers(raw: list) -> tuple[list[str], list[str]]:
    """Uppercase, de-dup, and drop anything that doesn't look like a ticker.
    Returns `(kept, dropped)` preserving first-seen order."""
    seen: set[str] = set()
    kept: list[str] = []
    dropped: list[str] = []
    for item in raw or []:
        s = str(item).strip().upper()
        if not s or s in seen:
            continue
        seen.add(s)
        if _TICKER_RE.match(s):
            kept.append(s)
        else:
            dropped.append(s)
    return kept, dropped


# --- Persistence ----------------------------------------------------------


def _save_to_disk(slug: str, thesis: Thesis) -> Path:
    path = THESES_DIR / f"{slug}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(thesis.model_dump_json(indent=2))
    return path


# --- Public API -----------------------------------------------------------


async def discover(
    *,
    topic: str | None = None,
    ticker: str | None = None,
    force_refresh: bool = False,
    ingest: bool = False,
    db_path: Path | None = None,
) -> DiscoveryResult:
    """Discover a grounded halo-graph thesis from a TOPIC or a seed TICKER.

    Exactly one of `topic` / `ticker` must be provided. Caches per input on
    disk (`theses/adhoc_{slug}.json`); pass `force_refresh=True` to rebuild.
    The universe is capped at `MAX_UNIVERSE`. `ingest=True` first downloads +
    embeds SEC filings for any universe ticker not yet in the manifest, so
    filings-grounding has a corpus to check (opt-in — a first ingest is minutes
    + embedding cost per ticker). `db_path` overrides the graph store (tests).

    `DiscoveryResult.error` is set (thesis None, nothing written) on invalid
    args, LLM failure, unparseable/vague proposal, or a grounded thesis that
    fails schema validation. Callers must check it.
    """
    if (topic and ticker) or (not topic and not ticker):
        return DiscoveryResult(
            slug="",
            thesis=None,
            path=Path(),
            error="exactly one of `topic` or `ticker` is required",
        )

    slug = adhoc_slug(topic=topic, ticker=ticker)
    cached_path = THESES_DIR / f"{slug}.json"
    if cached_path.exists() and not force_refresh:
        try:
            cached = Thesis.model_validate_json(cached_path.read_text())
            logger.info(f"[discovery] cache hit: {cached_path.name}")
            return DiscoveryResult(
                slug=slug,
                thesis=cached,
                path=cached_path,
                cached=True,
                n_universe=len(cached.universe),
                n_edges_grounded=len(cached.relationships),
            )
        except (ValidationError, json.JSONDecodeError, OSError) as e:
            logger.warning(f"[discovery] stale cache at {cached_path}: {e}; regenerating")

    # (A) Propose.
    try:
        parsed, _raw = await asyncio.to_thread(_propose, topic=topic, ticker=ticker)
    except Exception as e:  # noqa: BLE001 — surface as a result, never crash the caller
        logger.error(f"[discovery] proposal LLM call failed: {e}")
        return DiscoveryResult(
            slug=slug, thesis=None, path=cached_path, error=f"LLM proposal failed: {e}"
        )
    if not parsed or ("error" in parsed and "name" not in parsed):
        return DiscoveryResult(
            slug=slug,
            thesis=None,
            path=cached_path,
            error=str(parsed.get("error") or "LLM returned non-JSON / vague input"),
        )

    universe, dropped = _clean_tickers(parsed.get("universe", []))
    if not universe:
        return DiscoveryResult(
            slug=slug,
            thesis=None,
            path=cached_path,
            error="proposal contained no valid tickers",
            dropped_tickers=dropped,
        )
    universe_set = set(universe)
    # Ticker mode: enforce the contract that the seed is in the universe and
    # leads the anchors (the prompt asks for it, but the model may drop it).
    seed = ticker.strip().upper() if ticker else ""
    if seed and _TICKER_RE.match(seed) and seed not in universe_set:
        universe.insert(0, seed)
        universe_set.add(seed)
    anchors_clean, _ = _clean_tickers(parsed.get("anchor_tickers", []))
    anchors = [a for a in anchors_clean if a in universe_set] or universe[:1]
    if seed and seed in universe_set:
        anchors = [seed] + [a for a in anchors if a != seed]

    # Cap the universe (anchors kept first so they always survive the cut).
    if len(universe) > MAX_UNIVERSE:
        capped = anchors + [t for t in universe if t not in anchors]
        universe = capped[:MAX_UNIVERSE]
        universe_set = set(universe)

    # Optionally ingest filings for the (capped) universe so grounding can use
    # first-party SEC evidence instead of falling back to news co-mention alone.
    if ingest:
        await _ensure_ingested(universe)

    # (B) Ground each proposed edge against filings + news.
    name_map = await asyncio.to_thread(_name_map, universe)
    # Grounding runs sequentially (each edge awaited before the next), which is
    # why the shared `news_cache` is race-free; parallelising with gather()
    # later would need to synchronise it. `to_thread` keeps the blocking
    # filings/news I/O off the event loop.
    news_cache: dict[str, list] = {}
    now = datetime.now(UTC).isoformat()
    all_edges: list[GraphEdge] = []
    grounded_edges: list[GraphEdge] = []
    seen_pairs: set[tuple[str, str]] = set()
    for rel in parsed.get("relationships", []):
        frm = str(rel.get("from", "")).strip().upper()
        to = str(rel.get("to", "")).strip().upper()
        etype = str(rel.get("type", "peer")).strip().lower()
        if frm not in universe_set or to not in universe_set or frm == to:
            continue
        if (frm, to) in seen_pairs:  # graph_edges is UNIQUE(slug, from, to)
            continue
        seen_pairs.add((frm, to))
        if etype not in ("supplier", "customer", "peer", "competitor"):
            etype = "peer"
        try:
            confidence, evidence = await asyncio.to_thread(
                _ground_edge, frm, to, etype, name_map, news_cache
            )
        except Exception as e:  # noqa: BLE001 — an uncheckable edge is just ungrounded
            logger.warning(f"[discovery] grounding failed for {frm}->{to}: {e}")
            confidence, evidence = 0.0, []
        edge = GraphEdge(
            **{"from": frm},
            to=to,
            type=etype,  # type: ignore[arg-type]
            note=str(rel.get("note", "")),
            confidence=confidence,
            grounded=confidence >= GROUND_THRESHOLD,
            evidence=evidence,
            thesis_slug=slug,
            as_of=now,
        )
        all_edges.append(edge)
        if edge.grounded:
            grounded_edges.append(edge)

    # Assemble the grounded thesis — only corroborated edges survive.
    thesis_dict = dict(parsed)
    thesis_dict["universe"] = universe
    thesis_dict["anchor_tickers"] = anchors
    thesis_dict["relationships"] = [
        {"from": e.from_, "to": e.to, "type": e.type, "note": e.note} for e in grounded_edges
    ]
    try:
        thesis = Thesis.model_validate(thesis_dict)
    except ValidationError as e:
        logger.error(f"[discovery] grounded thesis failed validation for {slug}: {e}")
        return DiscoveryResult(
            slug=slug,
            thesis=None,
            path=cached_path,
            n_universe=len(universe),
            n_edges_proposed=len(all_edges),
            n_edges_grounded=len(grounded_edges),
            dropped_tickers=dropped,
            error=f"grounded thesis failed schema validation: {e.errors()[:2]}",
        )

    try:
        path = _save_to_disk(slug, thesis)
    except OSError as e:
        logger.error(f"[discovery] could not save thesis {slug}: {e}")
        return DiscoveryResult(
            slug=slug,
            thesis=None,
            path=cached_path,
            n_universe=len(universe),
            n_edges_proposed=len(all_edges),
            n_edges_grounded=len(grounded_edges),
            dropped_tickers=dropped,
            error=f"could not save thesis: {e}",
        )

    # Persist the full graph (grounded or not) for audit + traversal.
    nodes = [
        GraphNode(
            ticker=t,
            name=name_map.get(t, t),
            thesis_slug=slug,
            first_seen=now,
            last_seen=now,
        )
        for t in universe
    ]
    try:
        graph.save_graph(slug, nodes, all_edges, db_path=db_path)
    except Exception as e:  # noqa: BLE001 — thesis is already saved; graph is a sidecar
        logger.warning(f"[discovery] graph persist failed for {slug}: {e}")

    logger.info(
        f"[discovery] {slug}: universe={len(universe)} "
        f"edges {len(grounded_edges)}/{len(all_edges)} grounded → {path}"
    )
    return DiscoveryResult(
        slug=slug,
        thesis=thesis,
        path=path,
        n_universe=len(universe),
        n_edges_proposed=len(all_edges),
        n_edges_grounded=len(grounded_edges),
        dropped_tickers=dropped,
    )


# --- CLI ------------------------------------------------------------------


async def _cli(topic: str) -> None:
    result = await discover(topic=topic)
    if result.error:
        print(f"discovery failed: {result.error}", file=sys.stderr)
        sys.exit(1)
    print(
        json.dumps(
            {
                "slug": result.slug,
                "thesis_path": str(result.path),
                "n_universe": result.n_universe,
                "n_edges_proposed": result.n_edges_proposed,
                "n_edges_grounded": result.n_edges_grounded,
                "dropped_tickers": result.dropped_tickers,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python -m agents.discovery "<topic>"', file=sys.stderr)
        sys.exit(1)
    asyncio.run(_cli(" ".join(sys.argv[1:])))
