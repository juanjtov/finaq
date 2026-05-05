"""Telegram bot runner — Step 10b skeleton (allowlist + /help + /start).

This module builds a `python-telegram-bot.Application` configured with:
  - the FINAQ bot token (`TELEGRAM_BOT_TOKEN` from .env)
  - an allowlist of chat_ids (`TELEGRAM_CHAT_ID`, comma-separated for
    future multi-account, single value today)
  - the `/help` and `/start` command handlers (the welcome message)

Subsequent step files (10c–10g) layer in /drill, /scan, /status, /note,
/thesis, /analyze, and the natural-language fallback.

Why a separate module rather than inlining handlers in
`scripts/run_telegram_bot.py`:
  - Handlers are unit-testable in isolation (`tests/test_telegram.py`)
    without spinning up the long-poll loop.
  - `scripts/run_telegram_bot.py` stays a 10-line entrypoint that becomes
    a systemd unit in Step 12.
  - Other future code paths (e.g. Triage's "send alert" call in Step 11)
    can `from data.telegram import send_alert` without dragging in the
    bot-startup machinery.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging as _logging
import os
from datetime import UTC, datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Callable, Coroutine
from urllib.parse import urlencode

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from utils import logger

# Mute httpx INFO logging — it logs every request URL including the bot
# token (e.g. `POST https://api.telegram.org/bot<TOKEN>/getUpdates`),
# which lands in our log file and leaks the secret if logs are shared.
# WARNING level still surfaces 4xx/5xx errors we'd want to see.
_logging.getLogger("httpx").setLevel(_logging.WARNING)

# --- Allowlist --------------------------------------------------------------

# Loaded from `TELEGRAM_CHAT_ID` at `build_app()` time. Comma-separated to
# leave room for future multi-account use without a schema change. Today
# it carries exactly one ID — yours.
_allowed_chat_ids: set[int] = set()


def _parse_allowlist(raw: str | None) -> set[int]:
    """Parse a comma-separated string of chat_ids. Empty / whitespace
    entries are dropped; non-numeric entries raise a clear error so a
    typo in `.env` doesn't silently disable the allowlist."""
    if not raw:
        return set()
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            raise ValueError(
                f"TELEGRAM_CHAT_ID contains non-numeric entry {part!r} — "
                f"each comma-separated value must be a Telegram chat_id (integer)"
            )
    return out


def _is_allowed(update: Update) -> bool:
    chat = update.effective_chat
    return chat is not None and chat.id in _allowed_chat_ids


# --- HTML escape -----------------------------------------------------------
#
# We send replies with `parse_mode="HTML"` rather than the legacy Markdown
# mode. Markdown breaks on innocuous content (e.g. `nvda_halo` reads as an
# unterminated italic, and many ticker / signal strings carry `_` or `*`).
# HTML only needs `<`, `>`, `&` escaped — much more forgiving for user-
# generated and machine-generated content alike.

_HTML_ESCAPES = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}


def _h(text: str | None) -> str:
    """Escape text for safe interpolation into Telegram HTML messages."""
    if text is None:
        return ""
    out = str(text)
    for char, repl in _HTML_ESCAPES.items():
        out = out.replace(char, repl)
    return out


# --- Safe send (retry + disk fallback for long-running replies) -----------


_SEND_RETRY_DELAYS_S = (2, 5, 10)
"""Backoff between retry attempts. Total ~17s of patience covers brief
VPN reconnects, transient ISP/Telegram routing flaps, and DPI-driven
RST storms that resolve quickly. Beyond that the network is genuinely
gone and the on-disk fallback kicks in."""

_SEND_FALLBACK_DIR = Path("data_cache/telegram/pending")
"""Where we write a reply that couldn't be delivered after retries. The
user can recover the content here even if the bot can never reach
Telegram. Single-user system — no inbox semantics, just a paper trail."""


async def _send_safe(
    update: Update,
    text: str,
    *,
    label: str = "reply",
    **kwargs,
) -> bool:
    """Send `text` via `update.message.reply_text(...)` with retries on
    transient network errors. Returns True on delivery, False after
    final failure (in which case the body is written to
    `data_cache/telegram/pending/` so it isn't lost).

    `label` is a short string used in logs / fallback filenames so the
    operator knows which reply was lost (e.g. "drill-NVDA-summary").

    Use this for ANY reply that took meaningful work to compute — drill-in
    summaries especially. Short ack messages don't need the wrapper
    (they're cheap to retry by user-facing UX: just resend the command).
    """
    from telegram.error import NetworkError, TimedOut

    last_error: Exception | None = None
    for attempt, delay in enumerate(_SEND_RETRY_DELAYS_S, start=1):
        try:
            await update.message.reply_text(text, **kwargs)
            if attempt > 1:
                logger.info(
                    f"[telegram] {label} delivered on attempt {attempt}"
                )
            return True
        except (NetworkError, TimedOut) as e:
            last_error = e
            logger.warning(
                f"[telegram] {label} send attempt {attempt} failed "
                f"({type(e).__name__}): {e}; retrying in {delay}s"
            )
            await asyncio.sleep(delay)
        except Exception as e:
            # Non-network failure (e.g. BadRequest from malformed HTML).
            # Don't retry — fix the message, not the network.
            logger.error(
                f"[telegram] {label} send failed permanently "
                f"({type(e).__name__}): {e}"
            )
            last_error = e
            break

    # All retries exhausted (or non-retryable failure). Persist body so
    # the user can recover it from disk.
    try:
        _SEND_FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        path = _SEND_FALLBACK_DIR / f"{ts}__{label}.html"
        path.write_text(text)
        logger.error(
            f"[telegram] {label} could not be delivered "
            f"({type(last_error).__name__ if last_error else 'unknown'}); "
            f"body saved to {path}"
        )
    except Exception as e:  # pragma: no cover — disk write should not fail
        logger.error(f"[telegram] could not even write fallback file: {e}")
    return False


async def _send_mc_chart(
    update: Update,
    state: dict,
    *,
    label: str = "mc-chart",
) -> bool:
    """Render the MC histogram from `state.monte_carlo` and send it as a
    Telegram photo with a short caption. Returns True on delivery, False
    on render failure or persistent network failure.

    The chart is generated on-the-fly via `utils.charts.mc_histogram_to_bytes`
    using `resolve_mc_samples` to recover the sample array from the saved
    DCF percentiles (saved demos drop the 8k-sample array for size). Same
    fixed seed as the dashboard, so the bot's chart matches what the user
    sees if they tap through to Streamlit.
    """
    from telegram.error import NetworkError, TimedOut

    mc = state.get("monte_carlo") or {}
    if not mc:
        return False

    try:
        from utils.charts import mc_histogram_to_bytes, resolve_mc_samples

        samples = resolve_mc_samples(mc)
        if not samples:
            logger.info(f"[telegram] {label}: no MC samples to render")
            return False
        ticker = state.get("ticker") or "?"
        thesis = (state.get("thesis") or {}).get("name") or "?"
        title = f"{ticker} — Monte Carlo fair-value distribution"
        png_bytes = mc_histogram_to_bytes(
            samples,
            current_price=mc.get("current_price"),
            title=title,
        )
    except Exception as e:
        logger.warning(f"[telegram] {label}: chart render failed: {e}")
        return False

    dcf = mc.get("dcf") or {}
    p10, p50, p90 = dcf.get("p10"), dcf.get("p50"), dcf.get("p90")
    if all(v is not None for v in (p10, p50, p90)):
        caption = (
            f"📈 <b>{_h(ticker)}</b> · {_h(thesis)}\n"
            f"Bands: <code>${p10:,.0f}</code> → "
            f"<code>${p50:,.0f}</code> → <code>${p90:,.0f}</code>"
        )
    else:
        caption = f"📈 {_h(ticker)} · {_h(thesis)} — Monte Carlo distribution"

    last_error: Exception | None = None
    for attempt, delay in enumerate(_SEND_RETRY_DELAYS_S, start=1):
        try:
            # Telegram requires a fresh BytesIO per attempt — `send_photo`
            # consumes the stream; a retry needs a new pointer.
            photo = io.BytesIO(png_bytes)
            await update.message.reply_photo(
                photo=photo, caption=caption, parse_mode="HTML"
            )
            if attempt > 1:
                logger.info(f"[telegram] {label} delivered on attempt {attempt}")
            return True
        except (NetworkError, TimedOut) as e:
            last_error = e
            logger.warning(
                f"[telegram] {label} send attempt {attempt} failed "
                f"({type(e).__name__}); retrying in {delay}s"
            )
            await asyncio.sleep(delay)
        except Exception as e:
            logger.error(
                f"[telegram] {label} permanently failed "
                f"({type(e).__name__}): {e}"
            )
            last_error = e
            break

    # On final failure, write the PNG to the same fallback dir as text
    # replies so the user can recover it from disk.
    try:
        _SEND_FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        path = _SEND_FALLBACK_DIR / f"{ts}__{label}.png"
        path.write_bytes(png_bytes)
        logger.error(
            f"[telegram] {label} undeliverable ({type(last_error).__name__ if last_error else 'unknown'}); "
            f"PNG saved to {path}"
        )
    except Exception as e:  # pragma: no cover — disk write should not fail
        logger.error(f"[telegram] could not write fallback PNG: {e}")
    return False


HandlerFn = Callable[[Update, ContextTypes.DEFAULT_TYPE], Coroutine[None, None, None]]


def require_allowlist(handler: HandlerFn) -> HandlerFn:
    """Decorator: silently drop messages from chat_ids that aren't
    allowlisted. We log at INFO so a paper trail exists if someone finds
    the bot's @username; we don't reply (replying would confirm the bot
    is alive to the stranger).
    """

    @wraps(handler)
    async def wrapper(
        update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not _is_allowed(update):
            chat = update.effective_chat
            chat_id = chat.id if chat else "?"
            chat_user = (
                chat.username or chat.first_name if chat else "?"
            ) if chat else "?"
            logger.info(
                f"[telegram] dropping message from unallowed chat_id={chat_id} "
                f"({chat_user})"
            )
            return
        await handler(update, context)

    return wrapper


# --- Welcome / help handler ------------------------------------------------

_HELP_TEXT = (
    "👋 <b>FINAQ bot</b> — your equity research assistant\n"
    "\n"
    "📈 <b>Drill-ins</b>\n"
    "/drill TICKER [thesis]   Run a full drill-in (5 min)\n"
    "  e.g. <code>/drill NVDA</code> or <code>/drill AVGO ai_cake</code>\n"
    "/analyze TOPIC            Synthesize an ad-hoc thesis &amp; drill-in\n"
    "  e.g. <code>/analyze defense semis</code>\n"
    "\n"
    "📡 <b>Monitoring</b>\n"
    "/scan                     Surface alerts caught since the last check\n"
    "/theses                   List all available theses (slugs + universe)\n"
    "/thesis NAME              Show a thesis's tickers + recent activity\n"
    "/note TICKER text         Drop a note future Triage will weigh\n"
    "  e.g. <code>/note NVDA trim 20% if Q3 misses $42B</code>\n"
    "\n"
    "🔧 <b>System</b>\n"
    "/status                   Triage last-run, alerts in 24h, recent errors\n"
    "/help                     This message\n"
    "\n"
    "You can also just ask in plain text — \"what's NVDA looking like\" or "
    "\"analyze data center cooling\" — and I'll route it.\n"
)
"""Welcome / /help message rendered with `parse_mode='HTML'`. HTML mode is
preferred over Markdown because tickers, thesis slugs, and signal strings
routinely contain `_` characters that the legacy Markdown parser treats
as italic delimiters and fails when no closing `_` is found."""


@require_allowlist
async def help_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Reply with the welcome message. Bound to both `/help` and `/start`
    so users see the command list on first contact and can re-summon it
    any time."""
    await update.message.reply_text(_HELP_TEXT, parse_mode="HTML")


@require_allowlist
async def echo_placeholder(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Catch-all for unknown slash commands like `/foo`. Free text is now
    routed through `nl_fallback` (Step 10f) — this handler only fires
    when the user types a command we don't recognise."""
    text = update.message.text or ""
    await update.message.reply_text(
        f"❓ Unknown command: <code>{_h(text[:80])}</code>\n"
        f"Try /help for what's available, or just type plain text and "
        f"I'll route it.",
        parse_mode="HTML",
    )


# --- Natural-language fallback (Step 10f) ---------------------------------


@require_allowlist
async def nl_fallback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Route free-text messages through the Haiku intent classifier
    (`agents.router.classify`) and dispatch to the matching slash-command
    handler. When confidence < 0.7 OR intent is `unknown`, reply with a
    clarification prompt instead of guessing.

    Logs every decision so we can monitor whether the 0.7 threshold needs
    tuning (see `docs/POSTPONED.md §2 — Telegram NL router calibration`).
    """
    from types import SimpleNamespace as _SN

    text = (update.message.text or "").strip()
    if not text:
        return

    # Lazy import keeps bot startup snappy and avoids importing OpenRouter
    # client when the message handler is never called.
    from agents.router import classify, should_dispatch

    decision = await classify(text)
    dispatched = should_dispatch(decision)
    logger.info(
        f"[telegram-nl] intent={decision.intent} "
        f"args={decision.args} confidence={decision.confidence:.2f} "
        f"dispatched={dispatched} text={text[:120]!r}"
    )

    if not dispatched:
        # Either unknown intent or low confidence. Don't burn LLM credits
        # on the wrong action — ask for clarification instead. The exit
        # also fires for genuinely confused inputs ("hmm", "thanks").
        await update.message.reply_text(
            "❓ Not sure what you meant. Try one of:\n"
            "• <code>/drill TICKER</code> — full drill-in (5 min)\n"
            "• <code>/scan</code> — recent alerts\n"
            "• <code>/status</code> — system health\n"
            "• <code>/note TICKER text</code> — drop a note for Triage\n"
            "• <code>/thesis NAME</code> — a thesis's universe + activity\n"
            "• <code>/help</code> — full command list",
            parse_mode="HTML",
        )
        return

    intent = decision.intent
    args_dict = decision.args or {}

    if intent == "drill":
        ticker = (args_dict.get("ticker") or "").strip().upper()
        if not ticker:
            await update.message.reply_text(
                "I detected a drill request but couldn't pick out the ticker. "
                "Try <code>/drill TICKER</code>.",
                parse_mode="HTML",
            )
            return
        thesis = (args_dict.get("thesis") or "").strip()
        nl_args: list[str] = [ticker]
        if thesis:
            nl_args.append(thesis)
        await drill_command(update, _SN(args=nl_args))
        return

    if intent == "scan":
        await scan_command(update, _SN(args=[]))
        return

    if intent == "status":
        await status_command(update, _SN(args=[]))
        return

    if intent == "help":
        await help_command(update, _SN(args=[]))
        return

    if intent == "thesis":
        name = (args_dict.get("name") or "").strip()
        if not name:
            await update.message.reply_text(
                "I detected a thesis lookup but couldn't pick out the name. "
                "Try <code>/thesis NAME</code> or <code>/theses</code> to list.",
                parse_mode="HTML",
            )
            return
        await thesis_command(update, _SN(args=[name]))
        return

    if intent == "note":
        ticker = (args_dict.get("ticker") or "").strip().upper()
        note_text = (args_dict.get("text") or "").strip()
        if not ticker or not note_text:
            await update.message.reply_text(
                "I detected a note but couldn't extract both the ticker and "
                "the text. Try <code>/note TICKER your text here</code>.",
                parse_mode="HTML",
            )
            return
        # /note expects positional args: [ticker, *text_words]. Split the
        # text on whitespace so the existing handler reassembles it via
        # `' '.join(args[1:])`.
        await note_command(update, _SN(args=[ticker, *note_text.split()]))
        return

    if intent == "analyze":
        # Step 10e isn't built yet — same honest UX as the /drill inline
        # keyboard's "Synthesize custom" button.
        topic = (args_dict.get("topic") or "").strip()
        await update.message.reply_text(
            f"⏳ Custom-thesis synthesis (<code>/analyze</code>) lands in "
            f"<b>Step 10e</b> — not built yet.\n\n"
            f"For ad-hoc topics like <i>{_h(topic) or 'this'}</i>, two paths "
            f"available today:\n"
            f"• Drill against the Buffett-framework thesis: "
            f"<code>/drill TICKER general</code>\n"
            f"• Force a thematic thesis: "
            f"<code>/drill TICKER ai_cake</code> "
            f"(or <code>nvda_halo</code> / <code>construction</code>).",
            parse_mode="HTML",
        )
        return

    # Defensive fallback — should never hit unless the schema and handler
    # drift. Log loudly so we notice.
    logger.warning(
        f"[telegram-nl] unhandled intent {intent!r} after should_dispatch=True"
    )
    await update.message.reply_text(
        "❓ I understood your intent but don't have a handler for it yet. "
        "Try /help."
    )


# --- Thesis resolution ----------------------------------------------------

THESES_DIR = Path("theses")


def _list_thesis_slugs() -> list[str]:
    """Slugs of every JSON file in /theses/ — used to validate /drill args
    and to autodiscover which thesis a ticker belongs to."""
    if not THESES_DIR.exists():
        return []
    return sorted(p.stem for p in THESES_DIR.glob("*.json"))


def _resolve_thesis_slug(ticker: str, requested: str | None = None) -> str | None:
    """Pick a thesis slug for a `/drill TICKER [thesis]` invocation.

    Resolution order:
      1. If `requested` is given and is a valid slug, use it. (The user
         may want to force a thesis even if the ticker isn't in that
         thesis's universe — the dashboard banner handles that case.)
      2. If `requested` is given but doesn't match any thesis → return
         None so the caller can surface "thesis not found".
      3. If `requested` is None and the ticker is in some thesis's
         universe → return that thesis.
      4. If `requested` is None and the ticker is in NO universe →
         return None so the caller can suggest /analyze.

    Returns the slug, or None for cases (2) and (4) — caller distinguishes
    via `requested` to choose the right error message.
    """
    slugs = _list_thesis_slugs()
    if not slugs:
        return None
    if requested:
        norm = requested.strip().lower().replace("-", "_").replace(" ", "_")
        if norm in slugs:
            return norm
        # User typo — don't silently use a wrong slug; ask.
        return None
    ticker_upper = ticker.strip().upper()
    for slug in slugs:
        try:
            data = json.loads((THESES_DIR / f"{slug}.json").read_text())
        except Exception:
            continue
        universe = [str(t).upper() for t in (data.get("universe") or [])]
        if ticker_upper in universe:
            return slug
    # Ticker isn't in any thesis. Don't pick a random one — that runs an
    # expensive drill-in against a thesis the user didn't ask for. The
    # caller should suggest /analyze for ad-hoc topics.
    return None


# --- /drill ---------------------------------------------------------------


_DRILL_POLL_INTERVAL_S = 10
"""How often to check `is_running()` while a drill-in is in flight. 10s
keeps the load on the runner trivial; the user already saw a "running"
ack so they know it's working."""

_DRILL_MAX_WAIT_S = 600
"""Hard cap on the polling loop. Drill-ins target <5min per the spec, so
600s = 10min is a generous ceiling. If we hit it we still reply with a
'still running, will not block your chat' note rather than holding the
handler open indefinitely."""


def _build_streamlit_url(ticker: str, thesis_slug: str, run_id: str | None) -> str:
    """Build the dashboard URL with query params (Step 10g extends app.py
    to actually consume them). Falls back to localhost when the public
    URL isn't set in env (Step 12 sets `STREAMLIT_PUBLIC_URL`).

    Args are URL-encoded via urllib.parse.urlencode so a thesis name like
    "Halo · NVDA" can't sneak a `·` (or any other unsafe character) into
    the URL — Telegram's HTML parser refuses to make malformed URLs
    tappable, which the user observed first-hand.
    """
    base = os.environ.get("STREAMLIT_PUBLIC_URL", "http://localhost:8501").rstrip("/")
    params: dict[str, str] = {"ticker": ticker, "thesis": thesis_slug}
    if run_id:
        params["run_id"] = run_id
    return f"{base}/?{urlencode(params)}"


def _load_demo_state(ticker: str, thesis_slug: str, run_id: str) -> dict | None:
    """Read the saved drill-in state from `data_cache/demos/`. The runner
    writes `{TICKER}__{slug}__{run_id[:8]}.json` after the graph completes.
    """
    path = Path("data_cache/demos") / f"{ticker.upper()}__{thesis_slug}__{run_id[:8]}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        logger.warning(f"[telegram] could not read {path}: {e}")
        return None


def _valuation_verdict(p50: float | None, current_price: float | None) -> str:
    """Single-line classification of current price vs DCF P50. The bands
    are deliberately wide — the LLM-driven DCF carries enough noise that
    sub-5% differences are not actionable.

    Returned string already prefixed with ↗/↘/= so the user sees the
    direction at a glance.
    """
    if not p50 or not current_price or p50 <= 0:
        return "valuation: insufficient data"
    diff_pct = (current_price - p50) / p50 * 100
    abs_pct = abs(diff_pct)
    direction_word = "below" if diff_pct < 0 else "above"
    arrow = "↘" if diff_pct < 0 else ("↗" if diff_pct > 0 else "=")
    if abs_pct < 5:
        verdict = "fair value"
    elif diff_pct < -15:
        verdict = "meaningfully undervalued"
    elif diff_pct < -5:
        verdict = "moderately undervalued"
    elif diff_pct < 15:
        verdict = "moderately overvalued"
    else:
        verdict = "meaningfully overvalued"
    return f"{arrow} {abs_pct:.0f}% {direction_word} DCF P50 — {verdict}"


def _extract_action_first_sentence(report_md: str) -> str:
    """Pull the first sentence of the Action recommendation section so the
    Telegram summary stays compact. Returns empty string when absent."""
    if not report_md:
        return ""
    lines = report_md.splitlines()
    try:
        start = next(
            i for i, line in enumerate(lines) if line.startswith("## Action")
        )
    except StopIteration:
        return ""
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        if line.strip():
            body.append(line.strip())
    if not body:
        return ""
    text = " ".join(body)
    # First sentence — split on period followed by space or end.
    end = text.find(". ")
    if end > 0:
        return text[: end + 1]
    return text[:200]


def _format_drill_summary(state: dict, run_id: str, thesis_slug: str | None = None) -> str:
    """Build the ~10-line Telegram reply per the spec the user signed off on.
    Includes valuation verdict (over/under/fair vs DCF P50), one MC line
    with percentile bands + convergence ratio, top-1 risk, action sentence,
    and links to Streamlit + Notion. Rendered with `parse_mode='HTML'`.

    `thesis_slug` is the canonical slug from `/theses/{slug}.json` — the
    caller (drill_command) already resolved it. If absent we fall back to
    derivation from the thesis dict (legacy path used by the recovery
    script). Always pass slug explicitly when you have it.
    """
    ticker = state.get("ticker") or "?"
    thesis = state.get("thesis") or {}
    thesis_name = thesis.get("name") or "?"
    if not thesis_slug:
        thesis_slug = (state.get("thesis") or {}).get("slug") or _slug_from_state(state)
    confidence = state.get("synthesis_confidence") or "?"
    today = datetime.now(UTC).strftime("%Y-%m-%d")

    fund = state.get("fundamentals") or {}
    kpis = fund.get("kpis") or {}
    current_price = kpis.get("current_price")

    mc = state.get("monte_carlo") or {}
    dcf = mc.get("dcf") or {}
    p10, p50, p90 = dcf.get("p10"), dcf.get("p50"), dcf.get("p90")
    convergence = mc.get("convergence_ratio")

    risk = state.get("risk") or {}
    top_risks = risk.get("top_risks") or []

    action = _extract_action_first_sentence(state.get("report") or "")

    streamlit_url = _build_streamlit_url(ticker, thesis_slug, run_id)
    notion_url = state.get("notion_report_url") or ""

    lines: list[str] = []
    lines.append(f"<b>{_h(ticker)}</b> · {_h(thesis_name)} · {today}")
    lines.append(f"Confidence: <b>{_h(confidence)}</b>")
    lines.append("")

    if p50 and current_price:
        lines.append(f"📊 Valuation: {_h(_valuation_verdict(p50, current_price))}")
        lines.append(
            f"   Current <code>${current_price:,.2f}</code> vs DCF P50 "
            f"<code>${p50:,.2f}</code>"
        )
        lines.append("")

    if p10 and p50 and p90:
        line = (
            f"🎲 Monte Carlo: <code>${p10:,.0f}</code> → "
            f"<code>${p50:,.0f}</code> → <code>${p90:,.0f}</code> (P10–P50–P90)"
        )
        lines.append(line)
        if convergence is not None:
            agreement = (
                "DCF/multiple agree" if convergence >= 0.7
                else "DCF/multiple diverge"
            )
            lines.append(
                f"   Convergence <code>{convergence:.2f}</code> — {agreement}"
            )
        lines.append("")

    if top_risks:
        r = top_risks[0]
        title = r.get("title") or "(unnamed risk)"
        sev = r.get("severity") or "?"
        lines.append(f"⚠️ Top risk: {_h(title)} (sev {_h(str(sev))})")

    if action:
        lines.append(f"🎯 Action: {_h(action)}")

    if lines and lines[-1] != "":
        lines.append("")
    # Telegram (especially the iOS client) refuses to render
    # `http://localhost:...` as a tappable link because the host doesn't
    # resolve from the phone. When STREAMLIT_PUBLIC_URL isn't configured,
    # surface that explicitly so the user knows to open it from their Mac
    # rather than wondering why the "link" doesn't open. Step 12 sets
    # STREAMLIT_PUBLIC_URL on the droplet and this branch goes away.
    if "localhost" in streamlit_url or "127.0.0.1" in streamlit_url:
        lines.append(
            f'🔗 Streamlit (Mac only): <code>{_h(streamlit_url)}</code>'
        )
    else:
        lines.append(
            f'🔗 <a href="{_h(streamlit_url)}">Open in Streamlit</a>'
        )
    if notion_url:
        lines.append(f'🗒️ <a href="{_h(notion_url)}">View in Notion</a>')

    return "\n".join(lines)


def _slug_from_state(state: dict) -> str:
    """The thesis dict carried in state was loaded from a JSON file; the
    file's stem is the slug. We don't always have it on the dict, so as a
    fallback derive from the thesis name."""
    name = (state.get("thesis") or {}).get("name") or ""
    return name.lower().replace(" ", "_").replace("-", "_")


@require_allowlist
async def drill_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """`/drill TICKER [thesis]` — kick off a drill-in via ui/_runner and
    reply with the structured summary when it completes. Reuses the same
    daemon-thread runner the dashboard uses, so a Streamlit-side and a
    Telegram-side drill-in for the same (ticker, thesis) coalesce.

    When the ticker isn't in any pre-defined thesis universe AND no thesis
    was specified, we don't pick silently — we send an inline keyboard so
    the user explicitly chooses between the generic Buffett-framework
    thesis (cheap, fast) or synthesizing a custom thesis first
    (`/analyze`, ~$0.05 + 30s extra). See `_send_drill_choice_prompt` and
    `drill_choice_callback` for the keyboard plumbing.
    """
    args = context.args or []
    if not args:
        slugs = _list_thesis_slugs()
        thesis_list = "\n".join(
            f"  • <code>{_h(s)}</code>" for s in slugs
        ) if slugs else "  (none — populate /theses first)"
        await update.message.reply_text(
            "Usage: <code>/drill TICKER [thesis]</code>\n"
            "Examples: <code>/drill NVDA</code>, <code>/drill AVGO ai_cake</code>\n\n"
            "<b>Available theses:</b>\n"
            f"{thesis_list}\n\n"
            "Tip: omit the thesis arg and I'll auto-pick if the ticker is in "
            "any thesis universe — otherwise I'll ask you to choose.",
            parse_mode="HTML",
        )
        return
    ticker = args[0].strip().upper()
    requested_thesis = args[1] if len(args) > 1 else None
    thesis_slug = _resolve_thesis_slug(ticker, requested_thesis)

    if not thesis_slug and requested_thesis:
        # User specified a thesis but it didn't match any known slug.
        slugs = _list_thesis_slugs()
        slug_list = ", ".join(f"<code>{_h(s)}</code>" for s in slugs)
        body = (
            f"❓ Thesis <code>{_h(requested_thesis)}</code> not found.\n"
            f"Known theses: {slug_list}.\n"
            f"Try <code>/drill {_h(ticker)} {_h(slugs[0]) if slugs else 'ai_cake'}</code>."
        )
        await update.message.reply_text(body, parse_mode="HTML")
        return

    if not thesis_slug:
        # Ticker isn't in any thematic thesis universe. Don't pick silently
        # — ask the user via inline keyboard whether to use the generic
        # Buffett-framework thesis or synthesize a custom thesis first.
        await _send_drill_choice_prompt(update, ticker)
        return

    await _run_drill_and_reply(update.message, ticker, thesis_slug)


async def _run_drill_and_reply(
    reply_target,
    ticker: str,
    thesis_slug: str,
) -> None:
    """Kick off the drill, poll until completion, then send summary + chart.
    Used by both `drill_command` (slash path) and `drill_choice_callback`
    (inline-keyboard path).

    `reply_target` is a `telegram.Message` — both `update.message` (slash
    command) and `query.message` (callback query) satisfy this. We use
    `reply_target.reply_text(...)` for follow-ups so each message is
    threaded under the original `/drill`.
    """
    # Lazy import to avoid pulling Streamlit-side modules into the bot's
    # startup path. ui/_runner is the same code the dashboard uses.
    from ui import _runner

    ran_new = _runner.kick_off_drill(ticker, thesis_slug)
    if ran_new:
        ack = (
            f"🏃 Running drill-in for <b>{_h(ticker)}</b> × "
            f"<code>{_h(thesis_slug)}</code>. Takes ~5min — "
            f"I'll send the summary when it's done."
        )
    else:
        ack = (
            f"⏳ Drill-in already in flight for <b>{_h(ticker)}</b> × "
            f"<code>{_h(thesis_slug)}</code> — I'll send the summary when "
            f"it finishes."
        )
    await reply_target.reply_text(ack, parse_mode="HTML")

    # Poll. asyncio.sleep yields the event loop so other handlers can run.
    waited = 0.0
    while _runner.is_running(ticker, thesis_slug):
        await asyncio.sleep(_DRILL_POLL_INTERVAL_S)
        waited += _DRILL_POLL_INTERVAL_S
        if waited > _DRILL_MAX_WAIT_S:
            await reply_target.reply_text(
                f"⚠️ Still running after {_DRILL_MAX_WAIT_S // 60} min. "
                f"It'll keep going in the background; check Streamlit for "
                f"the result when it lands.",
            )
            return

    record = _runner.get_run_status(ticker, thesis_slug) or {}
    if record.get("error"):
        await reply_target.reply_text(
            f"❌ Drill-in failed: <code>{_h(record['error'])}</code>",
            parse_mode="HTML",
        )
        return
    run_id = record.get("run_id")
    if not run_id:
        await reply_target.reply_text(
            "⚠️ Drill-in completed but no run_id was recorded — check Mission Control."
        )
        return

    state = _load_demo_state(ticker, thesis_slug, run_id)
    if not state:
        await reply_target.reply_text(
            f"⚠️ Drill-in completed but the saved state at "
            f"<code>data_cache/demos/{_h(ticker)}__{_h(thesis_slug)}__{_h(run_id[:8])}.json</code> "
            f"is missing.",
            parse_mode="HTML",
        )
        return

    # Wrap reply_target into an Update-shaped object so the existing
    # `_send_safe(update, ...)` and `_send_mc_chart(update, ...)` paths
    # work without changing their signatures. Both helpers only access
    # `update.message.reply_text` / `update.message.reply_photo` — a
    # SimpleNamespace with `message=reply_target` satisfies that.
    from types import SimpleNamespace
    pseudo_update = SimpleNamespace(message=reply_target, effective_chat=None)

    summary = _format_drill_summary(state, run_id, thesis_slug=thesis_slug)
    await _send_safe(
        pseudo_update,
        summary,
        label=f"drill-{ticker}-{thesis_slug}-{run_id[:8]}",
        parse_mode="HTML",
        disable_web_page_preview=True,
    )

    # Also send the MC histogram as a photo so the user sees the
    # distribution shape (not just P10/P50/P90 numbers). Best-effort —
    # if the chart can't be rendered (e.g. MC was skipped) or the
    # network fails, the summary text alone is still enough.
    mc = state.get("monte_carlo") or {}
    if mc and mc.get("method") not in (None, "skipped"):
        await _send_mc_chart(
            pseudo_update,
            state,
            label=f"drill-{ticker}-{thesis_slug}-{run_id[:8]}-chart",
        )


# --- Inline-keyboard prompt for "ticker not in any thesis" ----------------

# Cost estimates printed on the buttons. Approximate based on Phase 0
# observed spend per drill-in (see Mission Control). Update if prices drift.
_COST_GENERIC = "~$0.50"
_COST_ANALYZE = "~$0.60"

_CALLBACK_PREFIX_GENERIC = "drill_generic"
_CALLBACK_PREFIX_ANALYZE = "drill_analyze"
_CALLBACK_PREFIX_CANCEL = "drill_cancel"


async def _send_drill_choice_prompt(update: Update, ticker: str) -> None:
    """Send the 3-button inline keyboard that lets the user choose how to
    drill on a ticker that isn't in any pre-built thesis."""
    keyboard = [
        [
            InlineKeyboardButton(
                f"Use generic thesis · {_COST_GENERIC}",
                callback_data=f"{_CALLBACK_PREFIX_GENERIC}:{ticker}",
            )
        ],
        [
            InlineKeyboardButton(
                f"Synthesize custom · {_COST_ANALYZE}",
                callback_data=f"{_CALLBACK_PREFIX_ANALYZE}:{ticker}",
            )
        ],
        [
            InlineKeyboardButton(
                "Cancel",
                callback_data=f"{_CALLBACK_PREFIX_CANCEL}:{ticker}",
            )
        ],
    ]
    body = (
        f"❓ <b>{_h(ticker)}</b> isn't in any pre-defined thesis "
        f"(<code>ai_cake</code>, <code>nvda_halo</code>, <code>construction</code>).\n\n"
        f"How do you want to proceed?\n"
        f"• <b>Use generic thesis</b> — runs against the Buffett-framework "
        f"<code>general</code> thesis (universal red flags: ROE, ROIC, debt/equity, "
        f"margin compression, accounting red flags).\n"
        f"• <b>Synthesize custom</b> — Sonnet builds a tailored thesis around "
        f"<code>{_h(ticker)}</code> first, then drills (Step 10e).\n"
        f"• <b>Cancel</b> — dismiss this prompt without drilling."
    )
    await update.message.reply_text(
        body,
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def drill_choice_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle taps on the /drill choice keyboard.

    The CallbackQueryHandler delivers `update.callback_query` rather than
    `update.message`, so we read the tap data, ack the spinner, and edit
    the original message in place to show what the bot is doing now.
    Allowlist enforced manually because the `@require_allowlist` decorator
    checks `update.message` which is None on callback queries.
    """
    query = update.callback_query
    if query is None:
        return
    chat = query.message.chat if query.message else None
    chat_id = chat.id if chat else None
    if chat_id is None or chat_id not in _allowed_chat_ids:
        # Drop silently — same posture as the message handler.
        if query:
            await query.answer()
        return

    # Always acknowledge the tap first (removes Telegram's "loading" spinner
    # on the button). Without this, the button shows a spinner indefinitely.
    await query.answer()

    data = query.data or ""
    if ":" not in data:
        return
    prefix, ticker = data.split(":", 1)
    ticker = ticker.strip().upper()

    if prefix == _CALLBACK_PREFIX_GENERIC:
        # User chose the generic thesis. Edit the prompt to reflect
        # the choice, then run the drill against the `general` slug.
        await query.edit_message_text(
            f"✓ Running drill on <b>{_h(ticker)}</b> with the "
            f"<code>general</code> Buffett-framework thesis…",
            parse_mode="HTML",
        )
        await _run_drill_and_reply(query.message, ticker, "general")
    elif prefix == _CALLBACK_PREFIX_ANALYZE:
        # /analyze isn't built yet (Step 10e). Be honest and point the
        # user at workable paths until then.
        await query.edit_message_text(
            f"⏳ Custom-thesis synthesis (<code>/analyze</code>) lands in "
            f"<b>Step 10e</b> — not built yet.\n\n"
            f"For now you can:\n"
            f"• Run <code>/drill {_h(ticker)} general</code> to use the "
            f"Buffett-framework thesis directly.\n"
            f"• Force a thematic thesis: "
            f"<code>/drill {_h(ticker)} ai_cake</code> "
            f"(or <code>nvda_halo</code> / <code>construction</code>).",
            parse_mode="HTML",
        )
    elif prefix == _CALLBACK_PREFIX_CANCEL:
        await query.edit_message_text(
            f"❌ Cancelled — no drill-in run for <code>{_h(ticker)}</code>.",
            parse_mode="HTML",
        )
    else:
        logger.warning(f"[telegram] unknown drill-choice callback: {data!r}")


# --- /scan ----------------------------------------------------------------


_TRIAGE_FIXTURE_PATH = Path("data_cache/fixtures/triage_alerts.json")


def _load_triage_alerts() -> list[dict]:
    """Until Step 11 lands real Triage, /scan reads the fixture file the
    Streamlit dashboard's "Run scan" button uses. Same shape, same source
    of truth — keeps the two surfaces aligned during the build."""
    if not _TRIAGE_FIXTURE_PATH.exists():
        return []
    try:
        return json.loads(_TRIAGE_FIXTURE_PATH.read_text())
    except Exception as e:
        logger.warning(f"[telegram] could not read triage fixture: {e}")
        return []


def _format_alerts(alerts: list[dict]) -> str:
    """HTML rendering of triage alerts, one block per alert with:
      - severity + ticker + thesis + signal headline
      - <i>Why it's an alert:</i> plain-English rationale
      - <i>Why it matters:</i> what to watch / position implication
      - 📊 Open in Streamlit (pre-loaded for ticker × thesis)
      - 🔗 Evidence URL

    Triage in Step 11 will populate `why_alert` and `why_attention`
    from the LLM. For Phase 0 / fixture-mode the strings are hand-written
    in `data_cache/fixtures/triage_alerts.json`. Older fixtures without
    those fields gracefully degrade to the signal line only.
    """
    if not alerts:
        return "✅ No alerts since the last scan."
    blocks = [f"📡 <b>{len(alerts)} alert(s)</b>"]
    for a in alerts[:20]:
        sev = a.get("severity", "?")
        ticker = a.get("ticker", "?")
        thesis = a.get("thesis", "?")
        signal = a.get("signal", "(no signal)")
        why_alert = (a.get("why_alert") or "").strip()
        why_attention = (a.get("why_attention") or "").strip()
        evidence_url = a.get("evidence_url")
        dashboard_url = _build_streamlit_url(ticker, thesis, run_id=None)

        parts: list[str] = []
        parts.append(
            f"<b>sev {_h(str(sev))}</b> <code>{_h(ticker)}</code> "
            f"({_h(thesis)})\n{_h(signal[:300])}"
        )
        if why_alert:
            parts.append(
                f"<i>Why it's an alert:</i> {_h(why_alert)}"
            )
        if why_attention:
            parts.append(
                f"<i>Why it matters:</i> {_h(why_attention)}"
            )
        link_row_parts: list[str] = []
        link_row_parts.append(
            f'📊 <a href="{_h(dashboard_url)}">Open dashboard</a>'
        )
        if evidence_url:
            link_row_parts.append(
                f'🔗 <a href="{_h(evidence_url)}">Evidence</a>'
            )
        parts.append(" · ".join(link_row_parts))
        blocks.append("\n".join(parts))
    if len(alerts) > 20:
        blocks.append(f"…and {len(alerts) - 20} more (truncated to 20).")
    # Blank line between blocks for readability — Telegram ignores most
    # whitespace but a real `\n\n` produces a visible gap.
    return "\n\n".join(blocks)


@require_allowlist
async def scan_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """`/scan` — surface the most recent triage alerts. Phase 0/Step 10c
    serves the fixture file; Step 11 swaps in `agents/triage.run()`."""
    alerts = _load_triage_alerts()
    body = _format_alerts(alerts)
    if alerts and "fixture" in (alerts[0].get("note") or "").lower():
        body += (
            "\n\n<i>Note: serving the Phase 0 fixture; real Triage lands in "
            "Step 11.</i>"
        )
    await update.message.reply_text(
        body, parse_mode="HTML", disable_web_page_preview=True
    )


# --- /status --------------------------------------------------------------


def _format_status_body() -> str:
    """Pull triage / alert / error / drill-in state from data/state.py and
    format it as a compact HTML status block. No external calls — purely
    SQLite reads, takes <50ms."""
    from data import state as _state

    now = datetime.now(UTC)
    cutoff_24h = (now - timedelta(hours=24)).isoformat()

    # Drill-in runs
    runs = _state.recent_runs(limit=5)
    if runs:
        last = runs[0]
        when = (last.get("started_at") or "?")[:16].replace("T", " ")
        last_drill_line = (
            f"Last drill-in: <code>{_h(last.get('ticker', '?'))}</code> × "
            f"<code>{_h(last.get('thesis', '?'))}</code> "
            f"({_h(last.get('status', '?'))}) at {_h(when)} UTC"
        )
    else:
        last_drill_line = "Last drill-in: never"

    # Triage runs
    triage = _state.recent_triage_runs(limit=1)
    if triage:
        t = triage[0]
        when = (t.get("ts") or "?")[:16].replace("T", " ")
        triage_line = (
            f"Last triage: {_h(str(t.get('alerts_emitted', 0)))} alerts / "
            f"{_h(str(t.get('items_scanned', 0)))} items at {_h(when)} UTC"
        )
    else:
        triage_line = "Last triage: never (Triage agent not yet running — Step 11)"

    # Alerts in 24h
    alerts = _state.recent_alerts(limit=200)
    in_24h = [a for a in alerts if (a.get("ts") or "") >= cutoff_24h]
    alerts_line = f"Alerts in last 24h: <b>{len(in_24h)}</b>"

    # Recent errors
    errors = _state.recent_errors(limit=10)
    errors_in_24h = [e for e in errors if (e.get("ts") or "") >= cutoff_24h]
    errors_line = f"Errors in last 24h: <b>{len(errors_in_24h)}</b>"
    if errors_in_24h:
        first_err = errors_in_24h[0]
        errors_line += (
            f" (most recent: <code>{_h(first_err.get('agent', '?'))}</code> — "
            f"{_h((first_err.get('message') or '')[:80])})"
        )

    return "\n".join([
        "🔧 <b>FINAQ status</b>",
        "",
        last_drill_line,
        triage_line,
        alerts_line,
        errors_line,
        "",
        "<i>Cost tracking lands when state.db schema gains the cost_usd column "
        "— see POSTPONED §1.</i>",
    ])


@require_allowlist
async def status_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """`/status` — system health summary read from `data_cache/state.db`."""
    body = _format_status_body()
    await update.message.reply_text(body, parse_mode="HTML")


# --- /note ----------------------------------------------------------------


@require_allowlist
async def note_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """`/note TICKER text` — append a free-text note to a thesis's Notion
    page. The note is timestamped + prefixed with the ticker so Triage
    (Step 11) can read the corpus and weight ongoing user concerns.

    Usage:
        /note NVDA trim 20% if Q3 misses $42B
        /note AAPL margin compression at risk on tariffs

    The thesis is auto-resolved from the ticker (same logic as /drill).
    Tickers not in any thesis attach to `general` so notes are never lost.
    """
    args = context.args or []
    if len(args) < 2:
        await update.message.reply_text(
            "Usage: <code>/note TICKER text</code>\n"
            "Example: <code>/note NVDA trim 20% if Q3 misses $42B</code>\n\n"
            "Notes attach to the ticker's resolved thesis page in Notion. "
            "Triage reads them next run.",
            parse_mode="HTML",
        )
        return

    ticker = args[0].strip().upper()
    text = " ".join(args[1:]).strip()
    if not text:
        await update.message.reply_text(
            "Note text is empty. Try again with content after the ticker.",
            parse_mode="HTML",
        )
        return

    # Resolve thesis. If ticker isn't in any thematic universe, attach to
    # `general` so the note is never silently dropped — same fallback the
    # interactive /drill prompt uses.
    thesis_slug = _resolve_thesis_slug(ticker, None)
    if not thesis_slug:
        slugs = _list_thesis_slugs()
        thesis_slug = "general" if "general" in slugs else (
            slugs[0] if slugs else None
        )
    if not thesis_slug:
        await update.message.reply_text(
            "❌ No theses configured. Add one to <code>/theses</code> first.",
            parse_mode="HTML",
        )
        return

    # Lazy-import notion to keep startup snappy and avoid a hard dep when
    # NOTION_API_KEY is absent.
    try:
        from data import notion as _notion
    except ImportError:
        await update.message.reply_text(
            "⚠️ Notion module not available. Note not saved.",
            parse_mode="HTML",
        )
        return

    if not _notion.is_configured():
        await update.message.reply_text(
            "⚠️ Notion isn't configured (NOTION_API_KEY missing). Note not saved.",
            parse_mode="HTML",
        )
        return

    ok = _notion.append_thesis_note(
        thesis_slug=thesis_slug, text=text, ticker=ticker
    )
    if ok:
        await update.message.reply_text(
            f"📝 Note attached to <code>{_h(thesis_slug)}</code> for "
            f"<b>{_h(ticker)}</b>:\n\n<i>{_h(text[:300])}</i>"
            + ("…" if len(text) > 300 else "")
            + "\n\nFuture <code>/scan</code> + Triage runs will weight this.",
            parse_mode="HTML",
        )
    else:
        await update.message.reply_text(
            f"⚠️ Couldn't attach note to <code>{_h(thesis_slug)}</code> — "
            f"check Notion connectivity or the bot logs.",
            parse_mode="HTML",
        )


# --- /thesis NAME ---------------------------------------------------------


@require_allowlist
async def thesis_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """`/thesis NAME` — show a thesis's universe + recent activity.

    Goes deeper than `/theses` (which lists slug + universe size for
    every thesis): for one thesis, also surfaces alerts in the last 7
    days (from state.db) and the most recent drill-in. Useful for
    "remind me what's in nvda_halo and what's been happening".
    """
    args = context.args or []
    if not args:
        await update.message.reply_text(
            "Usage: <code>/thesis NAME</code>\n"
            "Example: <code>/thesis ai_cake</code>\n\n"
            "Run <code>/theses</code> to list all available.",
            parse_mode="HTML",
        )
        return

    raw_name = " ".join(args).strip().lower().replace(" ", "_").replace("-", "_")
    slugs = _list_thesis_slugs()
    if raw_name not in slugs:
        slug_list = ", ".join(f"<code>{_h(s)}</code>" for s in slugs)
        await update.message.reply_text(
            f"❓ Thesis <code>{_h(raw_name)}</code> not found.\n"
            f"Known theses: {slug_list}.",
            parse_mode="HTML",
        )
        return

    try:
        thesis = json.loads((THESES_DIR / f"{raw_name}.json").read_text())
    except Exception as e:
        logger.warning(f"[telegram] /thesis {raw_name}: read failed: {e}")
        await update.message.reply_text(
            f"⚠️ Couldn't load <code>{_h(raw_name)}</code>: <code>{_h(str(e))}</code>",
            parse_mode="HTML",
        )
        return

    name = thesis.get("name", raw_name)
    summary = thesis.get("summary") or ""
    universe = thesis.get("universe") or []
    anchors = thesis.get("anchor_tickers") or []

    # Recent activity from state.db. Done lazily so a missing state.db
    # doesn't break the read-only paths.
    from data import state as _state

    cutoff_7d = (datetime.now(UTC) - timedelta(days=7)).isoformat()
    alerts = [
        a for a in _state.recent_alerts(limit=200)
        if a.get("thesis") == raw_name and (a.get("ts") or "") >= cutoff_7d
    ]
    runs = [r for r in _state.recent_runs(limit=20) if r.get("thesis") == raw_name]
    if runs:
        last = runs[0]
        last_when = (last.get("started_at") or "?")[:16].replace("T", " ")
        last_run_str = (
            f"<code>{_h(last.get('ticker', '?'))}</code> at "
            f"{_h(last_when)} UTC ({_h(last.get('status', '?'))})"
        )
    else:
        last_run_str = "<i>(no drill-ins yet)</i>"

    # Universe display: ⭐-prefix anchors so they're visually distinct.
    if universe:
        chip_lines = ", ".join(
            f"⭐ <code>{_h(t)}</code>" if t in anchors else f"<code>{_h(t)}</code>"
            for t in universe[:30]
        )
        if len(universe) > 30:
            chip_lines += f" …and {len(universe) - 30} more"
        universe_block = f"<b>Universe</b> ({len(universe)} tickers):\n{chip_lines}"
    else:
        universe_block = "<b>Universe:</b> <i>(empty — works for any ticker)</i>"

    # Trim summary so the reply fits Telegram's 4096-char limit comfortably
    # even when universe is large.
    summary_short = summary[:400] + ("…" if len(summary) > 400 else "")

    body = (
        f"📚 <b>{_h(name)}</b>\n"
        f"slug: <code>{_h(raw_name)}</code>\n\n"
        f"<i>{_h(summary_short)}</i>\n\n"
        f"{universe_block}\n\n"
        f"📡 Alerts (last 7d): <b>{len(alerts)}</b>\n"
        f"🏃 Last drill-in: {last_run_str}\n\n"
        f"Try: <code>/drill {anchors[0] if anchors else 'TICKER'} {raw_name}</code>"
    )
    await update.message.reply_text(
        body, parse_mode="HTML", disable_web_page_preview=True
    )


# --- /theses --------------------------------------------------------------


@require_allowlist
async def theses_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """`/theses` — list the thesis slugs available for `/drill TICKER thesis`.

    Shows each thesis's display name + universe size + anchor count so the
    user can pick the right one without leaving Telegram. Step 10d will
    add the deeper `/thesis NAME` command (universe + recent activity);
    this is the lightweight discovery surface for slug names only.
    """
    slugs = _list_thesis_slugs()
    if not slugs:
        await update.message.reply_text(
            "No theses found in <code>/theses</code>. Add a JSON file there "
            "to get started.",
            parse_mode="HTML",
        )
        return
    lines: list[str] = ["📚 <b>Available theses</b>", ""]
    for slug in slugs:
        try:
            data = json.loads((THESES_DIR / f"{slug}.json").read_text())
        except Exception as e:
            logger.warning(f"[telegram] could not read theses/{slug}.json: {e}")
            continue
        name = data.get("name", slug)
        universe = data.get("universe") or []
        anchors = data.get("anchor_tickers") or []
        if universe:
            usize = f"{len(universe)} ticker(s)"
            anchor_str = (
                f", anchors: {', '.join(anchors[:3])}"
                + ("…" if len(anchors) > 3 else "")
                if anchors
                else ""
            )
        else:
            usize = "any ticker (no universe)"
            anchor_str = ""
        lines.append(
            f"• <code>{_h(slug)}</code> — {_h(name)}\n"
            f"  <i>{usize}{anchor_str}</i>"
        )
    lines.append("")
    lines.append(
        "Run <code>/drill TICKER [thesis]</code> to use one. "
        "Without a thesis arg I'll auto-pick or ask."
    )
    await update.message.reply_text(
        "\n".join(lines), parse_mode="HTML", disable_web_page_preview=True
    )


# --- Bot bootstrapping ------------------------------------------------------


def build_app(*, token: str | None = None, allowlist: str | None = None) -> Application:
    """Build a configured `Application`. Reads token + allowlist from env
    by default; the kwargs let tests inject values without monkeypatching
    `os.environ`.

    Raises `RuntimeError` when token is missing — the bot can't start
    without it, and a clear error here beats an opaque telegram.error.

    Side effect: populates `_allowed_chat_ids` (module-level) from the
    parsed allowlist so handlers see the same set across the app lifetime.
    """
    token = (token or os.environ.get("TELEGRAM_BOT_TOKEN", "")).strip()
    allowlist = allowlist if allowlist is not None else os.environ.get(
        "TELEGRAM_CHAT_ID", ""
    )
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN not set — cannot start bot. Paste the token "
            "BotFather gave you into .env."
        )
    parsed = _parse_allowlist(allowlist)
    if not parsed:
        raise RuntimeError(
            "TELEGRAM_CHAT_ID not set or empty — refusing to start. The bot "
            "must allowlist at least one chat_id; without that anyone who "
            "guesses the @username can spend OpenRouter credits via /drill. "
            "Run `python -m scripts.discover_chat_id` to find your chat_id."
        )
    # Reset to a fresh set on every build — supports tests that build_app
    # multiple times with different allowlists, and avoids stale state on
    # bot restart.
    _allowed_chat_ids.clear()
    _allowed_chat_ids.update(parsed)

    app = Application.builder().token(token).build()
    # /help and /start both show the welcome message — Telegram users hit
    # /start automatically when they tap "Start" on a fresh chat with the
    # bot, so binding both keeps the first-contact flow smooth.
    app.add_handler(CommandHandler(["help", "start"], help_command))
    # Real slash commands.
    app.add_handler(CommandHandler("drill", drill_command))
    app.add_handler(CommandHandler("scan", scan_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("theses", theses_command))
    app.add_handler(CommandHandler("thesis", thesis_command))
    app.add_handler(CommandHandler("note", note_command))
    # Inline-keyboard callbacks for /drill's "ticker not in any thesis"
    # choice prompt. Must be registered before the catch-all handlers so
    # button taps reach this handler instead of the placeholder.
    app.add_handler(
        CallbackQueryHandler(
            drill_choice_callback,
            pattern=(
                f"^({_CALLBACK_PREFIX_GENERIC}|"
                f"{_CALLBACK_PREFIX_ANALYZE}|"
                f"{_CALLBACK_PREFIX_CANCEL}):"
            ),
        )
    )
    # Free text → Haiku-driven NL router → dispatch (Step 10f).
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, nl_fallback))
    # Unknown slash commands (e.g. `/foo`) → friendly error.
    app.add_handler(MessageHandler(filters.COMMAND, echo_placeholder))

    return app


def run() -> None:
    """Long-poll forever. Becomes a systemd unit in Step 12."""
    app = build_app()
    logger.info("[telegram] bot starting in long-poll mode")
    app.run_polling(drop_pending_updates=True)
