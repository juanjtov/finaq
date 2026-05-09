"""Ad-hoc thesis synthesizer (Step 10e — Discovery-lite).

Single LLM call that decomposes a free-text TOPIC (e.g. "defense semis")
or a single TICKER (e.g. "AAPL") into a `Thesis` JSON, validated against
`utils.schemas.Thesis`, and persisted to disk so the existing LangGraph
drill-in can run against it via `/drill TICKER {slug}`.

Persistence:
- Disk (always):  `theses/adhoc_{slug}.json` — same dir as curated theses
  so the existing runner / dashboard / `_list_thesis_slugs()` pick it up
  without changes. The `adhoc_` prefix lets you spot auto-generated theses
  in `/theses` and the dashboard sidebar.
- Notion (opt-in): if `NOTION_DB_ADHOC_THESES` is set in `.env`, mirror
  the JSON into that DB so the Phase 1 history is searchable. Best-effort
  — failures don't propagate.

Public API:
    await synthesize_adhoc_thesis(topic=None, ticker=None) -> AdhocThesisResult
        Synthesize, validate, persist, return slug + Thesis + filepath.

The Telegram `/analyze` handler and the `Synthesize custom` inline
keyboard branch both call this function — `topic` for the slash-command
path, `ticker` for the keyboard path (synthesize a single-name thesis
around the ticker the user already typed).

Why disk-first rather than Notion-first: the existing drill-in runner
(`ui/_runner.py`) reads thesis JSON from `theses/{slug}.json`. Saving to
disk first lets the rest of the system work unchanged. Notion is a
sidecar like the Reports / Watchlist DBs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from utils import logger
from utils.models import MODEL_ADHOC_THESIS
from utils.openrouter import get_client
from utils.schemas import Thesis

THESES_DIR = Path("theses")
ADHOC_PREFIX = "adhoc_"
"""Filename prefix for ad-hoc theses. Lets the dashboard sidebar +
/theses + /thesis distinguish them from curated theses (ai_cake etc.)
without a separate directory or schema field. Removable: `theses/adhoc_*.json`
can be deleted any time without breaking anything."""

LLM_MAX_TOKENS = 6000
"""Budget for the synthesis response. Bumped 2000 → 4000 → 6000 after
backtest "/analyze coursera education tech" truncated COUR's adhoc
thesis JSON at as_of=2025-09-05: the date-pinned mode + a longer
horizon prose drove the LLM past 4000 tokens before closing the final
brace, so the parser returned `{}` and the per-run JSON was empty.
6000 tokens now covers the prompt's worst-case (15 tickers,
8 relationships, 10 thresholds) with comfortable headroom."""

_RAW_FAIL_DIR = Path("data_cache/eval/adhoc_failures")
"""When the synthesizer's parse fails, we dump the raw LLM response here
so the user can grep / read it after the fact. Keeps the failure
debuggable without re-paying the cost just to see what came back."""

_PROMPT_PATH = Path(__file__).parent / "prompts" / "adhoc_thesis.md"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text()

# Slug normalisation: lower-case, alnum + underscores, max 40 chars (so
# the resulting filename `adhoc_{slug}.json` stays under most fs limits).
_SLUG_SAFE_RE = re.compile(r"[^a-z0-9_]+")


# --- Result type ----------------------------------------------------------


@dataclass
class AdhocThesisResult:
    """Return value from `synthesize_adhoc_thesis`. Carries everything
    the bot's `/analyze` handler needs to send a follow-up message and
    kick off a drill-in."""

    slug: str  # `adhoc_defense_semis` — used as `thesis_slug` for /drill
    thesis: Thesis  # validated Pydantic model
    path: Path  # disk path of the saved JSON
    notion_url: str | None = None
    cached: bool = False  # True when we returned an existing on-disk thesis
    error: str | None = None  # set on synthesis failure (e.g. LLM refusal)


# --- Slug derivation ------------------------------------------------------


def _slug_from_input(*, topic: str | None, ticker: str | None) -> str:
    """Build the `adhoc_{slug}` filename suffix from the user's input.

    Topic mode: lower-case the topic, replace whitespace + punctuation
    with underscores, truncate to 40 chars. "Defense Semis" → "defense_semis".

    Ticker mode: just lowercase the ticker. "AAPL" → "aapl".

    The `adhoc_` prefix is added by the caller — keep that out of the
    slug logic so a future migration to a different naming convention
    is a one-line change.
    """
    raw = (topic or ticker or "").strip().lower()
    raw = _SLUG_SAFE_RE.sub("_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw[:40] or "untitled"


def adhoc_slug(*, topic: str | None = None, ticker: str | None = None) -> str:
    """Public form of the slug derivation, including the `adhoc_` prefix.
    Exposed so the Telegram handler can pre-compute the slug before
    calling synthesize() (e.g. for cache-hit logging)."""
    return f"{ADHOC_PREFIX}{_slug_from_input(topic=topic, ticker=ticker)}"


# --- LLM call -------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        nl = text.find("\n")
        if nl > 0:
            text = text[nl + 1 :]
        if text.endswith("```"):
            text = text[:-3].rstrip()
    return text.strip()


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_thesis_response(raw: str) -> dict:
    """Parse the LLM response into a dict. Liberal — strips code fences,
    falls back to regex-extracting the first JSON object when the LLM
    wraps the JSON in prose. Logs at WARNING when both attempts fail
    so the operator can see the failure mode in the bot logs."""
    cleaned = _strip_code_fences(raw)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError as e:
        logger.info(
            f"[adhoc_thesis] strict json.loads failed ({e}); "
            f"trying regex fallback"
        )
    m = _JSON_OBJECT_RE.search(cleaned)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as e:
            logger.warning(
                f"[adhoc_thesis] regex-extracted JSON also failed to parse "
                f"({e}); raw len={len(raw)}, first 200: {raw[:200]!r}, "
                f"last 200: {raw[-200:]!r}"
            )
    else:
        logger.warning(
            f"[adhoc_thesis] no JSON object found in response; "
            f"raw len={len(raw)}, first 300: {raw[:300]!r}"
        )
    return {}


def _stash_failed_response(slug: str, raw: str) -> Path | None:
    """Save the raw LLM response to disk so the user can inspect it
    after the fact. Returns the saved path or None on disk-write failure.
    """
    try:
        _RAW_FAIL_DIR.mkdir(parents=True, exist_ok=True)
        from datetime import UTC, datetime

        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        path = _RAW_FAIL_DIR / f"{ts}__{slug}.txt"
        path.write_text(raw)
        logger.info(
            f"[adhoc_thesis] failed raw response stashed at {path} "
            f"({len(raw)} chars)"
        )
        return path
    except Exception as e:  # pragma: no cover — disk write should not fail
        logger.error(f"[adhoc_thesis] could not stash failed response: {e}")
        return None


def _build_user_prompt(*, topic: str | None, ticker: str | None) -> str:
    if ticker:
        return (
            f"Mode: TICKER\n"
            f"Input: {ticker.upper()}\n\n"
            f"Build a single-name thesis JSON around {ticker.upper()}. The "
            f"ticker MUST be the first anchor and present in the universe. "
            f"Surround it with 4-8 close peers / suppliers / customers."
        )
    return (
        f"Mode: TOPIC\n"
        f"Input: {topic}\n\n"
        f"Build a thesis JSON for this topic. Pick 5-15 representative "
        f"tickers with the 1-3 most pure-play names as anchors."
    )


def _call_llm(
    *,
    topic: str | None,
    ticker: str | None,
    as_of_date: str | None = None,
) -> tuple[dict, str]:
    """Single LLM call (model resolved via `MODEL_ADHOC_THESIS`).
    Returns `(parsed_dict, raw_response)` — the raw string is preserved
    so the caller can stash it on parse failure for offline inspection.
    Raises on network failure; caller wraps in try/except.

    Backtest mode (`as_of_date="YYYY-MM-DD"`): the as-of context block is
    prepended to the system prompt so the synthesizer doesn't shape the
    universe / material_thresholds with hindsight knowledge of post-as_of
    events. The model used (`MODEL_ADHOC_THESIS`) must have a training
    cutoff ≤ as_of_date — caller's responsibility to verify.
    """
    from utils.as_of import maybe_inject_as_of

    client = get_client()
    system = maybe_inject_as_of(_SYSTEM_PROMPT, as_of_date)
    resp = client.chat.completions.create(
        model=MODEL_ADHOC_THESIS,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": _build_user_prompt(topic=topic, ticker=ticker)},
        ],
        max_tokens=LLM_MAX_TOKENS,
    )
    raw = (resp.choices[0].message.content or "").strip()
    logger.info(
        f"[adhoc_thesis] LLM response received: {len(raw)} chars "
        f"(max_tokens={LLM_MAX_TOKENS})"
        f"{f' [as_of={as_of_date}]' if as_of_date else ''}"
    )
    return _parse_thesis_response(raw), raw


# --- Persistence ----------------------------------------------------------


def _save_to_disk(slug: str, thesis: Thesis, *, as_of_date: str | None = None) -> Path:
    """Write the validated thesis to disk.

    Production: `theses/{slug}.json` (next to curated theses).
    Backtest: `theses/backtest/{slug}__{as_of}.json` (segregated so
    production thesis listings aren't polluted by date-pinned variants).
    """
    if as_of_date:
        path = THESES_DIR / "backtest" / f"{slug}__{as_of_date}.json"
    else:
        path = THESES_DIR / f"{slug}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Pydantic's `.model_dump_json()` produces a stable, schema-valid
    # serialisation. Pretty-print with indent=2 so the file is
    # human-inspectable next to the curated theses.
    path.write_text(thesis.model_dump_json(indent=2))
    return path


def _maybe_save_to_notion(slug: str, thesis: Thesis) -> str | None:
    """If `NOTION_DB_ADHOC_THESES` is set, mirror the thesis into that
    DB. Returns the page URL or None on no-op / failure.

    Phase 0 / Step 10e doesn't auto-create the DB — the user runs
    bootstrap_notion to create it once. If the env var is missing, we
    skip silently. This keeps the synthesizer functional without Notion.
    """
    import os

    db_id = os.environ.get("NOTION_DB_ADHOC_THESES", "").strip()
    if not db_id:
        return None
    try:
        from data import notion as _notion
    except ImportError:
        return None
    if not _notion.is_configured():
        return None
    try:
        url = _notion.write_adhoc_thesis(
            slug=slug,
            thesis=thesis,
            db_id=db_id,
        )
        if url:
            logger.info(f"[adhoc_thesis] notion mirror persisted: {url}")
        return url
    except Exception as e:
        logger.warning(f"[adhoc_thesis] notion mirror failed: {e}")
        return None


# --- Public API -----------------------------------------------------------


async def synthesize_adhoc_thesis(
    *,
    topic: str | None = None,
    ticker: str | None = None,
    force_refresh: bool = False,
    as_of_date: str | None = None,
) -> AdhocThesisResult:
    """Synthesize an ad-hoc thesis. Caches per (topic|ticker) on disk so
    repeat invocations within a session don't re-pay the synthesis-LLM cost.

    Exactly one of `topic` or `ticker` must be provided.

    Returns an `AdhocThesisResult` whose `error` field is set on synthesis
    failure (vague input, LLM refused, JSON unparseable, schema validation
    failed). Callers should check `result.error` before dispatching a
    drill-in.

    Backtest mode (`as_of_date="YYYY-MM-DD"`): synthesizes the thesis with
    the as-of context block injected, and caches under
    `theses/backtest/adhoc_{slug}__{as_of}.json` so production adhoc theses
    aren't polluted by date-pinned variants.
    """
    if (topic and ticker) or (not topic and not ticker):
        return AdhocThesisResult(
            slug="",
            thesis=None,  # type: ignore[arg-type]
            path=Path(),
            error="exactly one of `topic` or `ticker` is required",
        )

    slug = adhoc_slug(topic=topic, ticker=ticker)
    if as_of_date:
        # Backtest cache: separate directory + as_of-keyed filename so
        # the production thesis list (`theses/*.json`) isn't polluted.
        cached_path = THESES_DIR / "backtest" / f"{slug}__{as_of_date}.json"
    else:
        cached_path = THESES_DIR / f"{slug}.json"

    # Cache hit — return the existing on-disk thesis unless the caller
    # explicitly asked for a refresh.
    if cached_path.exists() and not force_refresh:
        try:
            cached = Thesis.model_validate_json(cached_path.read_text())
            logger.info(f"[adhoc_thesis] cache hit: {cached_path.name}")
            return AdhocThesisResult(
                slug=slug, thesis=cached, path=cached_path, cached=True
            )
        except (ValidationError, json.JSONDecodeError) as e:
            logger.warning(
                f"[adhoc_thesis] stale cache at {cached_path}: {e}; "
                f"regenerating"
            )

    # Synthesize.
    try:
        parsed, raw_response = _call_llm(
            topic=topic, ticker=ticker, as_of_date=as_of_date,
        )
    except Exception as e:
        logger.error(f"[adhoc_thesis] LLM call failed: {e}")
        return AdhocThesisResult(
            slug=slug, thesis=None, path=cached_path,  # type: ignore[arg-type]
            error=f"LLM call failed: {e}",
        )

    if not parsed:
        # Couldn't parse a JSON object out of the response. Stash the raw
        # so the user can read what came back, and surface a 200-char
        # snippet in the error so the Telegram reply gives a hint.
        stashed = _stash_failed_response(slug, raw_response)
        snippet = raw_response[:200].replace("\n", " ")
        last = raw_response[-100:].replace("\n", " ") if len(raw_response) > 200 else ""
        # If the raw response looks JSON-ish (has `{`) but truncated,
        # tell the user that — actionable signal.
        truncated_hint = (
            " (looks like JSON-but-truncated; bump LLM_MAX_TOKENS or "
            "shorten topic)"
            if "{" in raw_response and not raw_response.rstrip().endswith("}")
            else ""
        )
        path_hint = (
            f"\n\nFull response saved to <code>{stashed}</code>"
            if stashed else ""
        )
        return AdhocThesisResult(
            slug=slug, thesis=None, path=cached_path,  # type: ignore[arg-type]
            error=(
                f"LLM returned non-JSON response{truncated_hint}. "
                f"First 200 chars: {snippet!r}. "
                f"Last 100 chars: {last!r}.{path_hint}"
            ),
        )
    if "error" in parsed and "name" not in parsed:
        return AdhocThesisResult(
            slug=slug, thesis=None, path=cached_path,  # type: ignore[arg-type]
            error=str(parsed.get("error") or "input too vague"),
        )

    try:
        thesis = Thesis.model_validate(parsed)
    except ValidationError as e:
        logger.error(f"[adhoc_thesis] schema validation failed for {slug}: {e}")
        # Stash the raw + parsed so the user can see what went wrong.
        _stash_failed_response(slug, raw_response)
        return AdhocThesisResult(
            slug=slug, thesis=None, path=cached_path,  # type: ignore[arg-type]
            error=f"thesis schema validation failed: {e.errors()[:2]}",
        )

    # Persist (disk + optional Notion mirror; Notion mirror skipped in
    # backtest mode — those are research artefacts, not user-facing).
    path = _save_to_disk(slug, thesis, as_of_date=as_of_date)
    notion_url = _maybe_save_to_notion(slug, thesis) if not as_of_date else None
    logger.info(f"[adhoc_thesis] synthesized + saved: {slug} → {path}")
    return AdhocThesisResult(
        slug=slug, thesis=thesis, path=path, notion_url=notion_url
    )
