"""Tests for `cio/notify.py` — Telegram formatter + send + Notion mirror.

The Telegram REST send is exercised against a stubbed httpx; we don't
hit `api.telegram.org`. Notion is monkey-patched.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cio import notify as cio_notify
from cio.planner import CIODecision, Plan


def _plan(*decisions: CIODecision, drills_capped: int = 0) -> Plan:
    return Plan(
        decisions=list(decisions), drill_budget=3, drills_capped=drills_capped,
    )


def _drill(ticker: str, thesis: str = "ai_cake", conf: str = "high") -> CIODecision:
    return CIODecision(
        action="drill", ticker=ticker, thesis=thesis,
        rationale=f"{ticker} drill rationale", confidence=conf,
    )


def _reuse(ticker: str, thesis: str = "ai_cake", run_id: str = "abc") -> CIODecision:
    return CIODecision(
        action="reuse", ticker=ticker, thesis=thesis,
        rationale=f"{ticker} still applies", confidence="medium",
        reuse_run_id=run_id,
    )


def _dismiss(ticker: str, thesis: str = "ai_cake") -> CIODecision:
    return CIODecision(
        action="dismiss", ticker=ticker, thesis=thesis,
        rationale=f"{ticker} nothing changed", confidence="low",
    )


# --- format_for_telegram --------------------------------------------------


def test_format_groups_by_action_with_emoji_headers():
    plan = _plan(_drill("NVDA"), _reuse("MSFT"), _dismiss("AAPL"))
    out = cio_notify.format_for_telegram(plan, trigger="heartbeat", duration_s=12.4)

    assert "📈" in out and "Drilled" in out
    assert "♻️" in out and "Reused" in out
    assert "🪦" in out and "Dismissed" in out
    assert "NVDA" in out and "MSFT" in out and "AAPL" in out
    assert "Duration: 12.4s" in out


def test_format_includes_drill_capped_note_when_nonzero():
    plan = _plan(_drill("NVDA"), drills_capped=2)
    out = cio_notify.format_for_telegram(plan, trigger="heartbeat", duration_s=1.0)
    assert "demoted 2" in out


def test_format_omits_capped_note_when_zero():
    plan = _plan(_drill("NVDA"))
    out = cio_notify.format_for_telegram(plan, trigger="heartbeat", duration_s=1.0)
    assert "demoted" not in out


def test_format_caps_dismiss_list_to_avoid_overflow():
    """Long dismiss runs get truncated with '… and N more dismissed.'"""
    decisions = [_dismiss(f"T{i}") for i in range(15)]
    plan = _plan(*decisions)
    out = cio_notify.format_for_telegram(plan, trigger="heartbeat", duration_s=1.0)
    assert "… and 7 more dismissed." in out


def test_format_html_escapes_user_content():
    """A rationale with `<` `>` `&` must be HTML-escaped or Telegram
    parse_mode='HTML' will reject the message."""
    d = CIODecision(
        action="drill", ticker="NVDA", thesis="ai_cake",
        rationale="margins < 30% & guide > $42B", confidence="high",
    )
    out = cio_notify.format_for_telegram(_plan(d), trigger="heartbeat", duration_s=1.0)
    assert "&lt;" in out and "&gt;" in out and "&amp;" in out
    assert "<" not in out.replace("</", "").replace("<a ", "").replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", "").replace("<code>", "").replace("</code>", "")


def test_format_with_only_dismissals_renders_nothing_for_other_groups():
    plan = _plan(_dismiss("NVDA"))
    out = cio_notify.format_for_telegram(plan, trigger="heartbeat", duration_s=1.0)
    assert "Drilled" not in out
    assert "Reused" not in out
    assert "Dismissed" in out


# --- _streamlit_base ------------------------------------------------------


def test_streamlit_base_rewrites_localhost_to_127(monkeypatch):
    monkeypatch.setenv("STREAMLIT_PUBLIC_URL", "http://localhost:8501")
    assert cio_notify._streamlit_base() == "http://127.0.0.1:8501"


def test_streamlit_base_strips_trailing_slash(monkeypatch):
    monkeypatch.setenv("STREAMLIT_PUBLIC_URL", "https://finaq.example.com/")
    assert cio_notify._streamlit_base() == "https://finaq.example.com"


def test_streamlit_base_default_when_unset(monkeypatch):
    monkeypatch.delenv("STREAMLIT_PUBLIC_URL", raising=False)
    out = cio_notify._streamlit_base()
    assert out.startswith("http://127.0.0.1")


# --- _first_chat_id -------------------------------------------------------


def test_first_chat_id_parses_single(monkeypatch):
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    assert cio_notify._first_chat_id() == 12345


def test_first_chat_id_parses_first_of_csv(monkeypatch):
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345, 67890")
    assert cio_notify._first_chat_id() == 12345


def test_first_chat_id_returns_none_when_missing(monkeypatch):
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert cio_notify._first_chat_id() is None


# --- send_telegram_message ------------------------------------------------


def test_send_telegram_skips_when_token_missing(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    assert cio_notify.send_telegram_message("hi") is False


def test_send_telegram_skips_when_chat_id_missing(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert cio_notify.send_telegram_message("hi") is False


def test_send_telegram_returns_true_on_200(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")

    captured: dict = {}

    def _fake_post(url, *, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return SimpleNamespace(status_code=200, text="ok")

    monkeypatch.setattr(cio_notify.httpx, "post", _fake_post)
    out = cio_notify.send_telegram_message("hello")
    assert out is True
    assert captured["url"].startswith("https://api.telegram.org/botfake-token/sendMessage")
    assert captured["json"]["chat_id"] == 12345
    assert captured["json"]["parse_mode"] == "HTML"
    assert captured["json"]["disable_web_page_preview"] is True


def test_send_telegram_returns_false_on_non_200(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")

    monkeypatch.setattr(
        cio_notify.httpx, "post",
        lambda *a, **k: SimpleNamespace(status_code=429, text="too many requests"),
    )
    assert cio_notify.send_telegram_message("hi") is False


def test_send_telegram_soft_fails_on_exception(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")

    def _boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(cio_notify.httpx, "post", _boom)
    assert cio_notify.send_telegram_message("hi") is False


# --- write_to_notion_alert ------------------------------------------------


def test_notion_mirror_no_op_when_db_unset(monkeypatch):
    monkeypatch.delenv("NOTION_DB_ALERTS", raising=False)
    plan = _plan(_drill("NVDA"))
    assert cio_notify.write_to_notion_alert(plan, trigger="heartbeat", summary="x") is None


def test_notion_mirror_passes_through_when_configured(monkeypatch):
    from data import notion as notion_mod

    monkeypatch.setenv("NOTION_DB_ALERTS", "fake-db-id")
    monkeypatch.setattr(notion_mod, "is_configured", lambda: True)
    monkeypatch.setattr(
        notion_mod, "write_alert",
        lambda **kw: ("alert-id-123", "https://notion.so/abc"),
    )
    plan = _plan(_drill("NVDA"))
    out = cio_notify.write_to_notion_alert(plan, trigger="heartbeat", summary="x")
    assert out == "https://notion.so/abc"


def test_notion_mirror_soft_fails_on_exception(monkeypatch):
    from data import notion as notion_mod

    monkeypatch.setenv("NOTION_DB_ALERTS", "fake-db-id")
    monkeypatch.setattr(notion_mod, "is_configured", lambda: True)

    def _boom(**kw):
        raise RuntimeError("notion 503")

    monkeypatch.setattr(notion_mod, "write_alert", _boom)
    plan = _plan(_drill("NVDA"))
    assert cio_notify.write_to_notion_alert(plan, trigger="heartbeat", summary="x") is None


# --- notify_cycle (top-level) --------------------------------------------


def test_notify_cycle_returns_status_dict(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    monkeypatch.setattr(
        cio_notify.httpx, "post",
        lambda *a, **k: SimpleNamespace(status_code=200, text="ok"),
    )
    monkeypatch.delenv("NOTION_DB_ALERTS", raising=False)

    plan = _plan(_drill("NVDA"), _dismiss("AAPL"))
    out = cio_notify.notify_cycle(plan, trigger="heartbeat", duration_s=5.0, summary_text="x")
    assert out["telegram_sent"] is True
    assert out["notion_url"] is None  # no NOTION_DB_ALERTS
    assert out["telegram_chars"] > 0


# --- _run_inspector_url + _drill_dashboard_url ---------------------------


def test_run_inspector_url_none_when_run_id_missing():
    assert cio_notify._run_inspector_url(None) is None
    assert cio_notify._run_inspector_url("") is None


def test_drill_dashboard_url_includes_query_params(monkeypatch):
    monkeypatch.setenv("STREAMLIT_PUBLIC_URL", "http://localhost:8501")
    url = cio_notify._drill_dashboard_url("NVDA", "ai_cake", "rid-1")
    assert "ticker=NVDA" in url
    assert "thesis=ai_cake" in url
    assert "run_id=rid-1" in url
    assert "127.0.0.1" in url  # localhost rewritten
