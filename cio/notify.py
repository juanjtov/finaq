"""CIO notify layer — Telegram + Notion sends.

Both targets are best-effort: a Telegram outage or Notion 503 must NOT
block the cycle (the cio_runs row is already persisted with the summary
locally). Failures are logged and surfaced as `False` from the helpers
so the dispatcher can record them but otherwise continue.

The Telegram path uses the Bot API's `sendMessage` endpoint directly
via httpx — the bot itself is purely message-handler-driven (long-poll
listener), so it has no built-in "push from server" hook. We bypass
python-telegram-bot for sends and just hit the REST API.
"""

from __future__ import annotations

import os
import urllib.parse
from typing import Any

import httpx

from cio.planner import Plan
from utils import logger

# Telegram message-length limit is 4096 chars. We aim shorter so
# truncation never bites; long rationales get clipped at 240 chars.
MAX_TELEGRAM_MSG = 3800
MAX_RATIONALE_CHARS = 240


def _h(text: str | None) -> str:
    """HTML-escape user content for Telegram parse_mode='HTML'."""
    if text is None:
        return ""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _streamlit_base() -> str:
    """Base URL for the dashboard. Honours `STREAMLIT_PUBLIC_URL` (the
    droplet path) and falls back to localhost-via-127.0.0.1 (Telegram
    silently strips `localhost` link entities — same fix we used for
    /drill replies)."""
    raw = os.environ.get("STREAMLIT_PUBLIC_URL", "").strip()
    if not raw:
        raw = "http://127.0.0.1:8501"
    return raw.rstrip("/").replace("://localhost", "://127.0.0.1")


def _run_inspector_url(run_id: str | None) -> str | None:
    """Deep-link to Run Inspector for a graph_runs row. Returns None when
    run_id is empty so the caller knows there's nothing to link to."""
    if not run_id:
        return None
    base = _streamlit_base()
    return f"{base}/run_inspector?run_id={urllib.parse.quote(run_id)}"


def _drill_dashboard_url(ticker: str, thesis: str | None, run_id: str | None) -> str:
    """Deep-link to the main dashboard pre-loaded with this drill."""
    base = _streamlit_base()
    qs = {"ticker": ticker.upper()}
    if thesis:
        qs["thesis"] = thesis
    if run_id:
        qs["run_id"] = run_id
    return f"{base}/?{urllib.parse.urlencode(qs)}"


# --- Telegram-formatted summary ------------------------------------------


def format_for_telegram(plan: Plan, *, trigger: str, duration_s: float) -> str:
    """HTML-formatted exec summary for Telegram. Mirrors the plain-text
    summary on the cio_runs row but with deep-links and richer typography.

    Length-bounded — we trim rationales to stay under MAX_TELEGRAM_MSG.
    The user sees the full rationale on the Mission Control page anyway.
    """
    header_emoji = {"heartbeat": "🩺", "catchup": "🔁", "on_demand": "🎯"}.get(trigger, "🤖")
    lines: list[str] = []
    lines.append(
        f"{header_emoji} <b>CIO {trigger.replace('_', ' ')}</b> — "
        f"{plan.n_drilled} drilled · {plan.n_reused} reused · "
        f"{plan.n_dismissed} dismissed"
    )
    if plan.drills_capped > 0:
        lines.append(
            f"<i>Drill budget cap demoted {plan.drills_capped} pair(s).</i>"
        )
    lines.append(f"<i>Duration: {duration_s:.1f}s</i>")
    lines.append("")

    # Group by action so the user reads drills first (loudest signal),
    # then reuses, then dismisses (quietest).
    drills = [d for d in plan.decisions if d.action == "drill"]
    reuses = [d for d in plan.decisions if d.action == "reuse"]
    dismisses = [d for d in plan.decisions if d.action == "dismiss"]

    if drills:
        lines.append("📈 <b>Drilled</b>")
        for d in drills:
            url = _drill_dashboard_url(d.ticker, d.thesis, None)
            thesis_part = f" / <code>{_h(d.thesis)}</code>" if d.thesis else ""
            rationale = _h((d.rationale or "")[:MAX_RATIONALE_CHARS])
            lines.append(
                f"• <a href=\"{url}\"><b>{_h(d.ticker)}</b></a>{thesis_part} "
                f"<i>({_h(d.confidence)})</i> — {rationale}"
            )
        lines.append("")

    if reuses:
        lines.append("♻️ <b>Reused</b>")
        for d in reuses:
            inspector = _run_inspector_url(d.reuse_run_id)
            link_part = (
                f"<a href=\"{inspector}\"><b>{_h(d.ticker)}</b></a>"
                if inspector
                else f"<b>{_h(d.ticker)}</b>"
            )
            thesis_part = f" / <code>{_h(d.thesis)}</code>" if d.thesis else ""
            rationale = _h((d.rationale or "")[:MAX_RATIONALE_CHARS])
            lines.append(
                f"• {link_part}{thesis_part} <i>({_h(d.confidence)})</i> — {rationale}"
            )
        lines.append("")

    if dismisses:
        lines.append("🪦 <b>Dismissed</b>")
        # Compact one-liner per dismiss — these are "nothing happened" signals.
        for d in dismisses[:8]:  # cap so the message stays readable
            thesis_part = f" / <code>{_h(d.thesis)}</code>" if d.thesis else ""
            rationale = _h((d.rationale or "")[:120])
            lines.append(
                f"• <b>{_h(d.ticker)}</b>{thesis_part} — {rationale}"
            )
        if len(dismisses) > 8:
            lines.append(f"<i>… and {len(dismisses) - 8} more dismissed.</i>")

    out = "\n".join(lines)
    if len(out) > MAX_TELEGRAM_MSG:
        out = out[: MAX_TELEGRAM_MSG - 80].rstrip() + "\n\n<i>(message trimmed)</i>"
    return out


# --- Telegram send (direct REST API) -------------------------------------


def _first_chat_id() -> int | None:
    """First allowlisted chat_id from TELEGRAM_CHAT_ID. We send the CIO
    summary to whichever chat is at index 0 — the user is the only
    allowlisted recipient by design."""
    raw = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    for tok in raw.split(","):
        tok = tok.strip()
        if tok.isdigit() or (tok.startswith("-") and tok[1:].isdigit()):
            try:
                return int(tok)
            except ValueError:
                continue
    return None


def send_telegram_message(
    text: str,
    *,
    chat_id: int | None = None,
    parse_mode: str = "HTML",
) -> bool:
    """POST to `api.telegram.org/bot{TOKEN}/sendMessage`. Soft-fail.

    Returns True on 200 OK, False otherwise. The bot.run_polling loop
    in `data/telegram.py` is unrelated — that's the inbound listener.
    Outbound pushes from the cron path bypass it via direct REST.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    cid = chat_id if chat_id is not None else _first_chat_id()
    if not token or cid is None:
        logger.warning(
            "[cio.notify] telegram send skipped: missing TELEGRAM_BOT_TOKEN or "
            "TELEGRAM_CHAT_ID"
        )
        return False
    try:
        resp = httpx.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": cid,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            },
            timeout=10.0,
        )
        if resp.status_code == 200:
            return True
        logger.warning(
            f"[cio.notify] telegram send returned {resp.status_code}: "
            f"{resp.text[:200]}"
        )
        return False
    except Exception as e:
        logger.warning(f"[cio.notify] telegram send failed: {e}")
        return False


# --- Notion mirror (best-effort) -----------------------------------------


def write_to_notion_alert(plan: Plan, *, trigger: str, summary: str) -> str | None:
    """Persist the CIO cycle as a Notion Alerts row. Returns the page URL
    on success, None on no-op / failure.

    No-op when:
      - NOTION_API_KEY missing
      - NOTION_DB_ALERTS missing
      - The notion-client SDK isn't installed

    The cio_runs row in state.db is the canonical record; the Notion row
    is just a convenience for users who run their journal in Notion.
    """
    if not os.environ.get("NOTION_DB_ALERTS"):
        return None
    try:
        from data import notion as _notion
    except ImportError:
        return None
    if not _notion.is_configured():
        return None
    try:
        # We piggy-back on the existing alerts schema. The CIO posts a
        # synthetic alert: ticker = "CIO", thesis = trigger, signal = first
        # decision's rationale, severity = 1 (informational).
        first = plan.decisions[0] if plan.decisions else None
        signal = (
            f"CIO {trigger} — {plan.n_drilled}d / {plan.n_reused}r / "
            f"{plan.n_dismissed}x — "
            f"{first.rationale if first else summary}"
        )[:500]
        result = _notion.write_alert(
            ticker="CIO",
            thesis_name=trigger,
            severity=1,
            signal=signal,
        )
        # write_alert returns (page_id, url) or None.
        if isinstance(result, tuple) and len(result) == 2:
            return result[1]
        return None
    except Exception as e:
        logger.warning(f"[cio.notify] notion mirror failed: {e}")
        return None


# --- Top-level entry -----------------------------------------------------


def notify_cycle(
    plan: Plan,
    *,
    trigger: str,
    duration_s: float,
    summary_text: str,
) -> dict:
    """Compose a Telegram message + send + best-effort Notion mirror.

    Returns a small dict with what landed where, useful for the
    dispatcher's exit-status log line and tests.
    """
    tg_text = format_for_telegram(plan, trigger=trigger, duration_s=duration_s)
    sent_telegram = send_telegram_message(tg_text)
    notion_url = write_to_notion_alert(plan, trigger=trigger, summary=summary_text)
    return {
        "telegram_sent": sent_telegram,
        "notion_url": notion_url,
        "telegram_chars": len(tg_text),
    }
