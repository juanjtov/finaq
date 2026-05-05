"""Tier 1 unit tests for data/telegram.py — bot skeleton + allowlist.

We don't spin up the long-poll loop or contact Telegram — handlers are
plain async coroutines that take an `Update` and a `Context`. Tests
construct fake `Update`s with the chat_id we want to assert against, then
call the handler and check what `reply_text` saw.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from data import telegram as tg


# --- Helpers ---------------------------------------------------------------


def _make_update(chat_id: int | None, text: str = "") -> SimpleNamespace:
    """Minimal Update-shaped object for handler tests. Carries effective_chat
    and message.reply_text mock so handlers can write back, plus message.text
    so the echo-placeholder handler can read what the user said."""
    reply_mock = AsyncMock()
    chat = (
        SimpleNamespace(id=chat_id, username="juan", first_name="Juan")
        if chat_id is not None
        else None
    )
    message = SimpleNamespace(reply_text=reply_mock, text=text)
    return SimpleNamespace(effective_chat=chat, message=message)


# --- Allowlist parsing ----------------------------------------------------


def test_parse_allowlist_single_id():
    assert tg._parse_allowlist("12345") == {12345}


def test_parse_allowlist_multiple_ids_comma_separated():
    assert tg._parse_allowlist("123,456, 789") == {123, 456, 789}


def test_parse_allowlist_empty_string_returns_empty_set():
    assert tg._parse_allowlist("") == set()
    assert tg._parse_allowlist(None) == set()


def test_parse_allowlist_drops_blank_entries():
    """Trailing commas / accidental double commas shouldn't blow up."""
    assert tg._parse_allowlist("123,,456,") == {123, 456}


def test_parse_allowlist_rejects_non_numeric():
    """A typo in .env mustn't silently produce an empty allowlist (which
    would lock you out of your own bot). Raise loudly instead."""
    with pytest.raises(ValueError, match="non-numeric"):
        tg._parse_allowlist("123,not-a-number,456")


# --- Allowlist enforcement (decorator) ------------------------------------


@pytest.mark.asyncio
async def test_require_allowlist_drops_unallowed_chat(monkeypatch):
    """A message from a non-allowlisted chat_id must not reach the
    decorated handler."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    inner = AsyncMock()
    decorated = tg.require_allowlist(inner)
    update = _make_update(chat_id=999)  # NOT in allowlist
    await decorated(update, SimpleNamespace())
    inner.assert_not_called()
    update.message.reply_text.assert_not_called(), (
        "must NOT reply — replying confirms bot existence to a stranger"
    )


@pytest.mark.asyncio
async def test_require_allowlist_passes_allowed_chat(monkeypatch):
    """A message from an allowlisted chat_id must reach the inner handler."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    inner = AsyncMock()
    decorated = tg.require_allowlist(inner)
    update = _make_update(chat_id=111)  # IN allowlist
    await decorated(update, SimpleNamespace())
    inner.assert_called_once()


@pytest.mark.asyncio
async def test_require_allowlist_handles_missing_chat(monkeypatch):
    """Defensive: weird Updates with no effective_chat (e.g. channel
    posts) must be dropped, not crash."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    inner = AsyncMock()
    decorated = tg.require_allowlist(inner)
    update = _make_update(chat_id=None)
    await decorated(update, SimpleNamespace())
    inner.assert_not_called()


# --- /help handler --------------------------------------------------------


@pytest.mark.asyncio
async def test_help_command_replies_with_welcome(monkeypatch):
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update = _make_update(chat_id=111, text="/help")
    await tg.help_command(update, SimpleNamespace())
    update.message.reply_text.assert_called_once()
    args, kwargs = update.message.reply_text.call_args
    body = args[0] if args else kwargs.get("text", "")
    assert "FINAQ bot" in body
    assert "/drill" in body
    assert "/analyze" in body
    assert "/scan" in body
    assert "/note" in body
    assert "/thesis" in body
    assert "/status" in body
    assert "/help" in body
    assert kwargs.get("parse_mode") == "HTML", (
        "must use HTML mode — Markdown breaks on `_` in tickers/slugs/signals"
    )


@pytest.mark.asyncio
async def test_help_command_dropped_for_unallowed(monkeypatch):
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update = _make_update(chat_id=999, text="/help")
    await tg.help_command(update, SimpleNamespace())
    update.message.reply_text.assert_not_called()


# --- Echo placeholder (Step 10b stand-in until 10c-10g land) -------------


@pytest.mark.asyncio
async def test_echo_placeholder_replies_with_under_construction(monkeypatch):
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update = _make_update(chat_id=111, text="/drill NVDA")
    await tg.echo_placeholder(update, SimpleNamespace())
    update.message.reply_text.assert_called_once()
    args, _ = update.message.reply_text.call_args
    assert "construction" in args[0].lower() or "/help" in args[0]


@pytest.mark.asyncio
async def test_echo_placeholder_dropped_for_unallowed(monkeypatch):
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update = _make_update(chat_id=999, text="hi")
    await tg.echo_placeholder(update, SimpleNamespace())
    update.message.reply_text.assert_not_called()


# --- build_app ------------------------------------------------------------


def test_build_app_raises_when_token_missing(monkeypatch):
    """Empty token + empty env var → fail loudly. Clear env so the local
    .env's real token doesn't satisfy the `token or os.environ.get(...)`
    fallback during dev runs."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
        tg.build_app(token="", allowlist="123")


def test_build_app_raises_when_allowlist_empty(monkeypatch):
    """Empty allowlist would leave the bot open to anyone — refuse to
    start rather than silently expose."""
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    with pytest.raises(RuntimeError, match="TELEGRAM_CHAT_ID"):
        tg.build_app(token="123:fake", allowlist="")


def test_build_app_populates_allowlist(monkeypatch):
    """build_app should mutate `_allowed_chat_ids` so subsequently-fired
    handlers see the right set."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", set(), raising=False)
    app = tg.build_app(token="999:fake-token-just-for-tests", allowlist="111,222")
    assert tg._allowed_chat_ids == {111, 222}
    # Also: the application has the right handler set so the bot
    # actually responds to /help and /start when started.
    handler_groups = list(app.handlers.values())
    assert handler_groups, "no handlers registered"
    handler_types = [type(h).__name__ for hs in handler_groups for h in hs]
    assert "CommandHandler" in handler_types
    assert "MessageHandler" in handler_types


def test_build_app_resets_allowlist_between_builds(monkeypatch):
    """A bot restart with a different .env should not retain stale IDs from
    the previous run. build_app() clears before populating."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {99999}, raising=False)
    tg.build_app(token="t:fake", allowlist="111")
    assert tg._allowed_chat_ids == {111}, (
        "stale 99999 should have been cleared before adding 111"
    )


# --- Valuation verdict (the hero line of /drill replies) ------------------


def test_valuation_verdict_fair_value_within_5_percent():
    """Within ±5% of P50 is 'fair value' — the LLM-driven DCF carries
    enough noise that sub-5% differences are not actionable."""
    assert "fair value" in tg._valuation_verdict(p50=200.0, current_price=205.0)


def test_valuation_verdict_meaningfully_undervalued_below_minus_15():
    out = tg._valuation_verdict(p50=200.0, current_price=150.0)
    assert "below" in out and "meaningfully undervalued" in out
    assert "↘" in out


def test_valuation_verdict_moderately_overvalued_5_to_15_above():
    out = tg._valuation_verdict(p50=200.0, current_price=220.0)
    assert "above" in out and "moderately overvalued" in out
    assert "↗" in out


def test_valuation_verdict_meaningfully_overvalued_above_15():
    out = tg._valuation_verdict(p50=200.0, current_price=300.0)
    assert "meaningfully overvalued" in out


def test_valuation_verdict_handles_missing_data():
    """Without P50 or price, the verdict is the literal 'insufficient data'
    string so the formatter doesn't blow up — and the user knows why."""
    assert tg._valuation_verdict(p50=None, current_price=200.0) == "valuation: insufficient data"
    assert tg._valuation_verdict(p50=200.0, current_price=None) == "valuation: insufficient data"
    assert tg._valuation_verdict(p50=0.0, current_price=200.0) == "valuation: insufficient data"


# --- Action-recommendation extraction -------------------------------------


def test_extract_action_first_sentence():
    md = (
        "## Top risks\n"
        "1. Risk one — sev 4\n\n"
        "## Action recommendation\n"
        "Hold existing exposure; add only on confirmation, not hope. "
        "Trim 20% if Q3 misses guide.\n\n"
        "## Watchlist\n- something\n"
    )
    sentence = tg._extract_action_first_sentence(md)
    assert sentence.startswith("Hold existing exposure")
    assert "Trim" not in sentence, "must stop at first period"


def test_extract_action_returns_empty_when_section_missing():
    md = "## Top risks\n1. Foo\n\n## Watchlist\n- thing"
    assert tg._extract_action_first_sentence(md) == ""


def test_extract_action_handles_empty_input():
    assert tg._extract_action_first_sentence("") == ""


# --- Summary formatter (the full /drill reply) ----------------------------


_DEMO_STATE_FIXTURE = {
    "ticker": "NVDA",
    "thesis": {"name": "AI cake", "slug": "ai_cake"},
    "synthesis_confidence": "medium",
    "fundamentals": {"kpis": {"current_price": 205.66}},
    "monte_carlo": {
        "dcf": {"p10": 90.74, "p50": 195.40, "p90": 280.22},
        "convergence_ratio": 0.78,
    },
    "risk": {
        "top_risks": [
            {
                "title": "AI Customer Concentration Risk",
                "severity": 4,
                "explanation": "Hyperscaler concentration",
                "sources": ["filings", "fundamentals"],
            }
        ]
    },
    "report": (
        "## Action recommendation\n"
        "Hold existing exposure and trim 20% only if Q3 misses guide.\n\n"
        "## Watchlist\n- foo\n"
    ),
    "notion_report_url": "https://notion.so/p/abc",
    "run_id": "abc12345-deadbeef",
}


def test_format_drill_summary_carries_all_required_sections(monkeypatch):
    monkeypatch.delenv("STREAMLIT_PUBLIC_URL", raising=False)
    out = tg._format_drill_summary(
        _DEMO_STATE_FIXTURE, run_id="abc12345-deadbeef", thesis_slug="ai_cake"
    )
    # Header — HTML <b> tags, ticker bolded
    assert "<b>NVDA</b>" in out
    assert "AI cake" in out
    # Confidence
    assert "Confidence: <b>medium</b>" in out
    # Valuation verdict line
    assert "Valuation:" in out
    assert "$205.66" in out and "$195.40" in out
    # MC line — `${p10:,.0f}` rounds 90.74 → "91", 195.40 → "195", 280.22 → "280"
    assert "Monte Carlo:" in out
    assert "$91" in out and "$195" in out and "$280" in out
    assert "Convergence" in out
    # Top risk
    assert "Top risk:" in out
    assert "Customer Concentration" in out
    assert "sev 4" in out
    # Action
    assert "Action:" in out
    assert "Hold existing exposure" in out
    # Links — Streamlit gets a `<code>` block in localhost mode (iOS
    # Telegram won't render http://localhost as a tappable link).
    # Ticker / thesis / run_id are still in the URL string.
    assert "ticker=NVDA" in out
    assert "thesis=ai_cake" in out
    assert "run_id=abc12345-deadbeef" in out
    assert "Streamlit (Mac only)" in out
    # Notion is a real public URL → tappable anchor tag.
    assert "<a href=" in out
    assert "View in Notion</a>" in out
    assert "https://notion.so/p/abc" in out


def test_format_drill_summary_renders_streamlit_as_link_when_public_url_set(
    monkeypatch,
):
    """When STREAMLIT_PUBLIC_URL is configured (Step 12 / droplet), the
    Streamlit URL should render as a tappable anchor — the localhost
    fallback was a workaround for iOS clients refusing to tap localhost."""
    monkeypatch.setenv("STREAMLIT_PUBLIC_URL", "https://finaq.example.com")
    out = tg._format_drill_summary(_DEMO_STATE_FIXTURE, run_id="abc", thesis_slug="ai_cake")
    assert "Open in Streamlit</a>" in out
    assert 'href="https://finaq.example.com/?ticker=NVDA' in out
    # No "Mac only" hint when the URL is publicly resolvable.
    assert "Mac only" not in out


def test_format_drill_summary_warns_about_localhost_fallback(monkeypatch):
    """Regression for the bug we observed live: an `http://localhost:...`
    URL inside an `<a href>` looks tappable to the formatter but iOS
    Telegram refuses to make it tappable from a phone, so the link
    appeared 'missing' to the user. The localhost branch now renders the
    URL as `<code>` text with a 'Mac only' hint instead — honest UX."""
    monkeypatch.delenv("STREAMLIT_PUBLIC_URL", raising=False)
    out = tg._format_drill_summary(_DEMO_STATE_FIXTURE, run_id="abc", thesis_slug="ai_cake")
    # Localhost URL should NOT be inside an <a> tag — the tag is the lie
    # iOS won't render. It IS inside <code> so the user can copy-paste.
    assert "<a href=\"http://localhost" not in out
    assert "<code>http://localhost:8501" in out
    assert "Mac only" in out


def test_format_drill_summary_does_not_emit_legacy_markdown_syntax(monkeypatch):
    """Regression guard: a future edit must not reintroduce `*bold*` or
    `_italic_` Markdown — those break on tickers/slugs containing `_`.
    The output is rendered as HTML, so these characters appearing as
    literal output (not inside `<code>`) would suggest someone copied
    the old Markdown formatter back in."""
    monkeypatch.delenv("STREAMLIT_PUBLIC_URL", raising=False)
    out = tg._format_drill_summary(_DEMO_STATE_FIXTURE, run_id="abc", thesis_slug="ai_cake")
    # `**bold**` Markdown is the most distinctive smell
    assert "**" not in out
    # Single-asterisk Markdown italic-or-bold around words
    import re
    assert not re.search(r"\*\w", out), (
        "found a Markdown-style asterisk-prefixed word — switch to HTML <b>"
    )


def test_format_drill_summary_uses_explicit_slug_not_derived_from_name(monkeypatch):
    """Regression guard for the bug we hit live: the formatter used to
    derive the slug from the thesis NAME ("Halo · NVDA"), producing
    "halo_·_nvda" — a `·` in the URL prevents Telegram from rendering the
    link as tappable. Now the slug is passed in by the caller (drill_command)
    so the link is always correct."""
    monkeypatch.delenv("STREAMLIT_PUBLIC_URL", raising=False)
    state = {
        **_DEMO_STATE_FIXTURE,
        "thesis": {"name": "Halo · NVDA"},  # name with `·` would break URL
    }
    out = tg._format_drill_summary(state, run_id="abc", thesis_slug="nvda_halo")
    assert "thesis=nvda_halo" in out
    assert "halo_·_nvda" not in out
    # Localhost rendering is in `<code>` rather than `<a>` (iOS won't make
    # localhost tappable anyway). Extract every URL substring and assert
    # the `·` doesn't leak in.
    import re
    urls = re.findall(r'(?:href="([^"]+)"|<code>([^<]+)</code>)', out)
    flattened = [u for href, code in urls for u in (href, code) if u]
    streamlit_urls = [u for u in flattened if "ticker=" in u]
    assert streamlit_urls, "expected at least one streamlit URL with ticker= param"
    for u in streamlit_urls:
        assert "·" not in u, f"malformed URL contains `·`: {u}"


def test_build_streamlit_url_url_encodes_special_chars(monkeypatch):
    """Defensive: even if a thesis slug somehow contained a special char
    (shouldn't happen given /theses/ filenames are snake_case), the URL
    must still be a valid URL — urlencode handles it."""
    monkeypatch.setenv("STREAMLIT_PUBLIC_URL", "https://finaq.example.com")
    url = tg._build_streamlit_url("NVDA", "halo · nvda", "abc 123")
    # Spaces become `+` (urlencode default), `·` becomes `%C2%B7`
    assert "halo+%C2%B7+nvda" in url or "halo%20%C2%B7%20nvda" in url
    assert "abc+123" in url or "abc%20123" in url


def test_format_drill_summary_falls_back_to_localhost_streamlit(monkeypatch):
    """When STREAMLIT_PUBLIC_URL isn't set (Phase 0 / 10c — Step 12 will
    set it on the droplet), links use localhost. This is a Mac-only useful
    link for now, which we accept until reachability ships."""
    monkeypatch.delenv("STREAMLIT_PUBLIC_URL", raising=False)
    out = tg._format_drill_summary(_DEMO_STATE_FIXTURE, run_id="abc12345")
    assert "http://localhost:8501" in out


def test_format_drill_summary_uses_public_url_when_set(monkeypatch):
    monkeypatch.setenv("STREAMLIT_PUBLIC_URL", "https://finaq.example.com")
    out = tg._format_drill_summary(_DEMO_STATE_FIXTURE, run_id="abc")
    assert "https://finaq.example.com/?ticker=NVDA" in out


def test_format_drill_summary_omits_notion_when_url_missing(monkeypatch):
    monkeypatch.delenv("STREAMLIT_PUBLIC_URL", raising=False)
    state = {**_DEMO_STATE_FIXTURE, "notion_report_url": ""}
    out = tg._format_drill_summary(state, run_id="abc", thesis_slug="ai_cake")
    assert "View in Notion" not in out
    # Streamlit URL is still present — just rendered as <code> in the
    # localhost branch since iOS won't tap localhost links.
    assert "Streamlit" in out
    assert "ticker=NVDA" in out


def test_format_drill_summary_handles_missing_mc(monkeypatch):
    """Drill-ins that skip MC (e.g. CRDO without shares_outstanding) must
    still produce a usable summary — we just drop the MC + valuation lines."""
    monkeypatch.delenv("STREAMLIT_PUBLIC_URL", raising=False)
    state = {**_DEMO_STATE_FIXTURE, "monte_carlo": {}}
    out = tg._format_drill_summary(state, run_id="abc")
    assert "Monte Carlo:" not in out
    assert "Valuation:" not in out
    # Risk + action survive even when MC is missing.
    assert "Top risk:" in out
    assert "Action:" in out


# --- Thesis resolution ----------------------------------------------------


def test_list_thesis_slugs_returns_existing_files():
    slugs = tg._list_thesis_slugs()
    # The repo has these three theses checked in — guard against a
    # silent rename / removal.
    assert "ai_cake" in slugs
    assert "construction" in slugs
    assert "nvda_halo" in slugs


def test_resolve_thesis_slug_uses_explicit_when_valid():
    assert tg._resolve_thesis_slug("NVDA", "ai_cake") == "ai_cake"


def test_resolve_thesis_slug_normalizes_user_typing():
    """User-typed thesis names should match case- and dash-insensitive."""
    # Passing the exact slug works (we already test that). Whitespace /
    # dashes / case in user-typed args also resolve.
    assert tg._resolve_thesis_slug("NVDA", "AI_CAKE") == "ai_cake"
    assert tg._resolve_thesis_slug("NVDA", "ai-cake") == "ai_cake"


def test_resolve_thesis_slug_returns_none_for_unknown_thesis():
    """A typo on the thesis arg should return None so the dispatcher can
    ask the user to pick from the known list — silently using the wrong
    thesis would waste a 5-min drill-in."""
    assert tg._resolve_thesis_slug("NVDA", "fake_thesis") is None


def test_resolve_thesis_slug_picks_universe_match():
    """No thesis specified → pick the first thesis whose universe contains
    the ticker. NVDA is in ai_cake's universe (and nvda_halo's anchor list)."""
    slug = tg._resolve_thesis_slug("NVDA", None)
    assert slug in ("ai_cake", "nvda_halo"), (
        f"NVDA should resolve to ai_cake or nvda_halo, got {slug}"
    )


def test_resolve_thesis_slug_returns_none_for_ticker_in_no_universe():
    """An off-the-radar ticker (e.g. AAPL — not in any of our 3 theses)
    should return None rather than silently fall through to the first
    thesis. The drill_command surfaces /analyze as the right tool for
    ad-hoc topics."""
    assert tg._resolve_thesis_slug("AAPL", None) is None
    # Sanity check that the same ticker DOES resolve when explicitly forced
    assert tg._resolve_thesis_slug("AAPL", "ai_cake") == "ai_cake"


# --- Streamlit URL builder ------------------------------------------------


def test_build_streamlit_url_carries_query_params(monkeypatch):
    monkeypatch.setenv("STREAMLIT_PUBLIC_URL", "https://example.com")
    url = tg._build_streamlit_url("NVDA", "ai_cake", "deadbeef-1234")
    assert url == "https://example.com/?ticker=NVDA&thesis=ai_cake&run_id=deadbeef-1234"


def test_build_streamlit_url_omits_run_id_when_absent(monkeypatch):
    monkeypatch.setenv("STREAMLIT_PUBLIC_URL", "https://example.com/")  # trailing slash
    url = tg._build_streamlit_url("NVDA", "ai_cake", None)
    assert url == "https://example.com/?ticker=NVDA&thesis=ai_cake"


# --- /scan handler --------------------------------------------------------


def test_format_alerts_handles_empty_list():
    out = tg._format_alerts([])
    assert "No alerts" in out


def test_format_alerts_renders_severity_and_signal():
    alerts = [
        {
            "ticker": "NVDA",
            "thesis": "ai_cake",
            "severity": 4,
            "signal": "filing_mentions: capacity constraint",
            "evidence_url": "https://sec.gov/...",
        },
        {
            "ticker": "MSFT",
            "thesis": "ai_cake",
            "severity": 3,
            "signal": "ai_capex_guidance: raised",
        },
    ]
    out = tg._format_alerts(alerts)
    assert "2 alert(s)" in out
    assert "NVDA" in out and "MSFT" in out
    assert "sev 4" in out and "sev 3" in out
    # Both alerts should get a dashboard link, the first should also get
    # an evidence link.
    assert "Open dashboard" in out
    assert "Evidence" in out
    assert "capacity constraint" in out


def test_format_alerts_renders_why_fields_when_present(monkeypatch):
    """Triage will populate why_alert / why_attention; Phase 0 fixture has
    them too. The renderer must surface them as labeled lines so the user
    sees the rationale, not just the raw signal string."""
    monkeypatch.delenv("STREAMLIT_PUBLIC_URL", raising=False)
    alerts = [
        {
            "ticker": "NVDA",
            "thesis": "ai_cake",
            "severity": 4,
            "signal": "filing_mentions: capacity constraint",
            "why_alert": "10-Q surfaces capacity-constraint language",
            "why_attention": "Capacity-bound revenue compresses upside",
        }
    ]
    out = tg._format_alerts(alerts)
    assert "Why it's an alert" in out
    assert "10-Q surfaces" in out
    assert "Why it matters" in out
    assert "Capacity-bound" in out


def test_format_alerts_gracefully_omits_missing_why_fields(monkeypatch):
    """Old fixtures (or alerts that lacked LLM rationale) should still
    render — we just drop the rationale lines."""
    monkeypatch.delenv("STREAMLIT_PUBLIC_URL", raising=False)
    alerts = [
        {
            "ticker": "NVDA",
            "thesis": "ai_cake",
            "severity": 4,
            "signal": "test signal",
        }
    ]
    out = tg._format_alerts(alerts)
    assert "Why it's an alert" not in out
    assert "Why it matters" not in out
    assert "NVDA" in out


def test_format_alerts_includes_per_alert_dashboard_link(monkeypatch):
    """Each alert should carry a dashboard link pre-loaded for that ticker
    × thesis — the user wants to drill in immediately if the alert looks
    interesting, without having to retype /drill. The URL is HTML-escaped
    (`&` → `&amp;`) because we render via parse_mode=HTML."""
    monkeypatch.setenv("STREAMLIT_PUBLIC_URL", "https://finaq.example.com")
    alerts = [
        {"ticker": "NVDA", "thesis": "ai_cake", "severity": 4, "signal": "x"},
        {"ticker": "MSFT", "thesis": "ai_cake", "severity": 3, "signal": "y"},
    ]
    out = tg._format_alerts(alerts)
    assert "https://finaq.example.com/?ticker=NVDA&amp;thesis=ai_cake" in out
    assert "https://finaq.example.com/?ticker=MSFT&amp;thesis=ai_cake" in out


def test_format_alerts_truncates_at_twenty():
    alerts = [
        {"ticker": f"T{i}", "thesis": "x", "severity": 1, "signal": "s"}
        for i in range(25)
    ]
    out = tg._format_alerts(alerts)
    assert "5 more" in out  # 25 - 20 = 5 hidden


@pytest.mark.asyncio
async def test_scan_command_replies_with_alerts(monkeypatch):
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    monkeypatch.setattr(
        tg,
        "_load_triage_alerts",
        lambda: [
            {
                "ticker": "NVDA",
                "thesis": "ai_cake",
                "severity": 4,
                "signal": "test signal",
            }
        ],
    )
    update = _make_update(chat_id=111, text="/scan")
    await tg.scan_command(update, SimpleNamespace())
    update.message.reply_text.assert_called_once()
    args, _ = update.message.reply_text.call_args
    assert "NVDA" in args[0]
    assert "test signal" in args[0]


# --- /status handler ------------------------------------------------------


@pytest.mark.asyncio
async def test_status_command_replies_with_status_summary(monkeypatch):
    """The /status handler reads from data/state.py — the conftest already
    redirects DB_PATH to a tmp file, so we just exercise the path on an
    empty DB and assert the structure."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update = _make_update(chat_id=111, text="/status")
    await tg.status_command(update, SimpleNamespace())
    update.message.reply_text.assert_called_once()
    args, kwargs = update.message.reply_text.call_args
    body = args[0]
    assert "FINAQ status" in body
    # Empty DB → all four lines surface "never" or "0".
    assert "Last drill-in:" in body
    assert "Last triage:" in body
    assert "Alerts in last 24h:" in body
    assert "Errors in last 24h:" in body


def test_status_body_reflects_recent_runs(monkeypatch, tmp_path):
    """Populate state.db with one drill-in run and confirm /status surfaces
    it in the 'Last drill-in' line."""
    from data import state as st

    db = tmp_path / "state.db"
    monkeypatch.setattr(st, "DB_PATH", db, raising=False)
    run_id = st.start_graph_run(ticker="NVDA", thesis="ai_cake")
    st.finish_graph_run(run_id, status="completed", confidence="medium")
    body = tg._format_status_body()
    assert "NVDA" in body
    assert "ai_cake" in body
    assert "completed" in body


# --- /drill handler — usage + thesis resolution branches ------------------


@pytest.mark.asyncio
async def test_drill_command_no_args_replies_with_usage_and_thesis_list(monkeypatch):
    """User typed /drill alone — show usage AND list all known thesis
    slugs so they don't have to remember them. Without this the user has
    to bail out and run /theses or /help."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    monkeypatch.setattr(
        tg, "_list_thesis_slugs", lambda: ["ai_cake", "construction", "general", "nvda_halo"]
    )
    update = _make_update(chat_id=111, text="/drill")
    context = SimpleNamespace(args=[])
    await tg.drill_command(update, context)
    update.message.reply_text.assert_called_once()
    args, _ = update.message.reply_text.call_args
    body = args[0]
    assert "Usage" in body and "/drill" in body
    # Available theses must be listed so the user can pick one.
    assert "Available theses" in body
    assert "ai_cake" in body
    assert "construction" in body
    assert "general" in body
    assert "nvda_halo" in body


# --- /theses --------------------------------------------------------------


@pytest.mark.asyncio
async def test_theses_command_lists_all_with_universe_size(monkeypatch):
    """`/theses` must list every thesis with display name + universe size.
    Empty-universe theses (e.g. general) are flagged as 'any ticker'."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update = _make_update(chat_id=111, text="/theses")
    await tg.theses_command(update, SimpleNamespace())
    update.message.reply_text.assert_called_once()
    args, kwargs = update.message.reply_text.call_args
    body = args[0]
    # Display names + slugs
    assert "Available theses" in body
    assert "ai_cake" in body
    assert "construction" in body
    assert "general" in body
    assert "nvda_halo" in body
    # General has empty universe → must be labeled clearly so the user
    # knows it works for any ticker.
    assert "any ticker" in body
    # Thematic theses have non-empty universe → ticker count shows up.
    assert "ticker(s)" in body
    assert kwargs.get("parse_mode") == "HTML"


@pytest.mark.asyncio
async def test_theses_command_dropped_for_unallowed(monkeypatch):
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update = _make_update(chat_id=999, text="/theses")
    await tg.theses_command(update, SimpleNamespace())
    update.message.reply_text.assert_not_called()


# --- general thesis MC inputs (regression) --------------------------------


def test_general_thesis_has_valuation_block_for_mc():
    """The MC engine refuses to run without a `valuation` block on the
    thesis — we observed this live when /drill WEN against general silently
    skipped MC. Pin that the general thesis carries the required fields."""
    raw = json.loads(Path("theses/general.json").read_text())
    val = raw.get("valuation") or {}
    assert "equity_risk_premium" in val
    assert "terminal_growth_rate" in val
    assert "discount_rate_floor" in val
    assert "discount_rate_cap" in val
    # Sanity bounds — Buffett-conservative defaults
    assert 0.04 <= val["equity_risk_premium"] <= 0.10
    assert 0.015 <= val["terminal_growth_rate"] <= 0.04


# --- /note ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_note_command_no_args_replies_with_usage(monkeypatch):
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update = _make_update(chat_id=111, text="/note")
    context = SimpleNamespace(args=[])
    await tg.note_command(update, context)
    update.message.reply_text.assert_called_once()
    args, _ = update.message.reply_text.call_args
    assert "Usage" in args[0] and "/note" in args[0]


@pytest.mark.asyncio
async def test_note_command_only_ticker_no_text_replies_with_usage(monkeypatch):
    """`/note NVDA` (ticker but no body) should be rejected. Without this
    a typo could create empty Notion paragraphs the user has to clean up."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update = _make_update(chat_id=111, text="/note NVDA")
    context = SimpleNamespace(args=["NVDA"])
    await tg.note_command(update, context)
    args, _ = update.message.reply_text.call_args
    assert "Usage" in args[0] or "empty" in args[0].lower()


@pytest.mark.asyncio
async def test_note_command_appends_to_resolved_thesis(monkeypatch):
    """`/note NVDA "trim 20%"` resolves NVDA → ai_cake (NVDA's anchor) and
    appends the note to ai_cake's Notion page.

    Uses `monkeypatch.setattr(data.notion, ...)` rather than replacing
    `sys.modules["data.notion"]` — the latter pattern leaks across tests
    because pytest doesn't restore sys.modules between cases.
    """
    from data import notion as nm

    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    monkeypatch.setattr(tg, "_resolve_thesis_slug", lambda t, r: "ai_cake")
    monkeypatch.setattr(nm, "is_configured", lambda: True)

    captured: dict = {}

    def _stub_append(*, thesis_slug, text, ticker=None):
        captured["thesis_slug"] = thesis_slug
        captured["text"] = text
        captured["ticker"] = ticker
        return True

    monkeypatch.setattr(nm, "append_thesis_note", _stub_append)

    update = _make_update(chat_id=111, text="/note NVDA trim 20% if Q3 misses")
    context = SimpleNamespace(args=["NVDA", "trim", "20%", "if", "Q3", "misses"])
    await tg.note_command(update, context)
    assert captured.get("thesis_slug") == "ai_cake"
    assert captured.get("ticker") == "NVDA"
    assert captured.get("text") == "trim 20% if Q3 misses"
    args, _ = update.message.reply_text.call_args
    assert "📝" in args[0] or "attached" in args[0]


@pytest.mark.asyncio
async def test_note_command_falls_back_to_general_for_unknown_ticker(monkeypatch):
    """When ticker isn't in any thematic universe, /note attaches to general
    rather than dropping the note silently."""
    from data import notion as nm

    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    monkeypatch.setattr(tg, "_resolve_thesis_slug", lambda t, r: None)
    monkeypatch.setattr(
        tg, "_list_thesis_slugs", lambda: ["ai_cake", "construction", "general", "nvda_halo"]
    )
    monkeypatch.setattr(nm, "is_configured", lambda: True)

    captured: dict = {}

    def _stub_append(*, thesis_slug, text, ticker=None):
        captured["thesis_slug"] = thesis_slug
        return True

    monkeypatch.setattr(nm, "append_thesis_note", _stub_append)

    update = _make_update(chat_id=111, text="/note AAPL margin compression risk")
    context = SimpleNamespace(args=["AAPL", "margin", "compression", "risk"])
    await tg.note_command(update, context)
    assert captured.get("thesis_slug") == "general", (
        "ad-hoc tickers must route to general so notes are never lost"
    )


@pytest.mark.asyncio
async def test_note_command_handles_notion_unconfigured(monkeypatch):
    from data import notion as nm

    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    monkeypatch.setattr(tg, "_resolve_thesis_slug", lambda t, r: "ai_cake")
    monkeypatch.setattr(nm, "is_configured", lambda: False)

    update = _make_update(chat_id=111, text="/note NVDA test")
    context = SimpleNamespace(args=["NVDA", "test"])
    await tg.note_command(update, context)
    args, _ = update.message.reply_text.call_args
    assert "Notion" in args[0] and "configured" in args[0].lower()


@pytest.mark.asyncio
async def test_note_command_dropped_for_unallowed(monkeypatch):
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update = _make_update(chat_id=999, text="/note NVDA test")
    context = SimpleNamespace(args=["NVDA", "test"])
    await tg.note_command(update, context)
    update.message.reply_text.assert_not_called()


# --- /thesis NAME --------------------------------------------------------


@pytest.mark.asyncio
async def test_thesis_command_no_args_replies_with_usage(monkeypatch):
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update = _make_update(chat_id=111, text="/thesis")
    context = SimpleNamespace(args=[])
    await tg.thesis_command(update, context)
    args, _ = update.message.reply_text.call_args
    assert "Usage" in args[0] and "/thesis" in args[0]


@pytest.mark.asyncio
async def test_thesis_command_unknown_name_lists_available(monkeypatch):
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update = _make_update(chat_id=111, text="/thesis fake")
    context = SimpleNamespace(args=["fake"])
    await tg.thesis_command(update, context)
    args, _ = update.message.reply_text.call_args
    body = args[0]
    assert "not found" in body.lower()
    # All real theses should be listed for navigation.
    assert "ai_cake" in body
    assert "general" in body


@pytest.mark.asyncio
async def test_thesis_command_renders_universe_and_anchors(monkeypatch):
    """/thesis ai_cake shows the universe with anchors prefixed by ⭐."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update = _make_update(chat_id=111, text="/thesis ai_cake")
    context = SimpleNamespace(args=["ai_cake"])
    await tg.thesis_command(update, context)
    args, kwargs = update.message.reply_text.call_args
    body = args[0]
    # Slug + display name surface
    assert "ai_cake" in body
    # Universe + anchor markers
    assert "Universe" in body
    assert "⭐" in body  # at least one anchor must be marked
    # Activity rows
    assert "Alerts" in body
    assert "Last drill-in" in body
    assert kwargs.get("parse_mode") == "HTML"


@pytest.mark.asyncio
async def test_thesis_command_normalizes_whitespace_and_dashes(monkeypatch):
    """User types '/thesis ai cake' or '/thesis ai-cake' — both should
    resolve to ai_cake (case + space + dash insensitive)."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update = _make_update(chat_id=111, text="/thesis AI CAKE")
    context = SimpleNamespace(args=["AI", "CAKE"])
    await tg.thesis_command(update, context)
    args, _ = update.message.reply_text.call_args
    # No "not found" — the resolver normalised "AI CAKE" → "ai_cake".
    assert "not found" not in args[0].lower()


@pytest.mark.asyncio
async def test_thesis_command_renders_general_with_empty_universe(monkeypatch):
    """The general thesis has empty universe — must say so explicitly so
    the user understands it's the catch-all rather than a misconfigured one."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update = _make_update(chat_id=111, text="/thesis general")
    context = SimpleNamespace(args=["general"])
    await tg.thesis_command(update, context)
    args, _ = update.message.reply_text.call_args
    body = args[0]
    assert "general" in body
    assert "any ticker" in body or "empty" in body


@pytest.mark.asyncio
async def test_thesis_command_dropped_for_unallowed(monkeypatch):
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update = _make_update(chat_id=999, text="/thesis ai_cake")
    context = SimpleNamespace(args=["ai_cake"])
    await tg.thesis_command(update, context)
    update.message.reply_text.assert_not_called()


# --- NL fallback (Step 10f) -----------------------------------------------


def _stub_router_decision(monkeypatch, *, intent, args=None, confidence=0.9):
    """Replace agents.router.classify with a stub returning a fixed
    RouterDecision. Avoids a real LLM call so NL tests are fast + offline."""
    from utils.schemas import RouterDecision

    decision = RouterDecision(
        intent=intent, args=args or {}, confidence=confidence
    )

    async def _classify(text):
        return decision

    from agents import router as r
    monkeypatch.setattr(r, "classify", _classify)
    return decision


@pytest.mark.asyncio
async def test_nl_fallback_low_confidence_replies_with_clarification(monkeypatch):
    """Below 0.7 confidence the bot must NOT dispatch — instead reply with
    the clarification menu so the user picks the right command."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    _stub_router_decision(monkeypatch, intent="unknown", confidence=0.1)

    update = _make_update(chat_id=111, text="hmm")
    await tg.nl_fallback(update, SimpleNamespace())
    args, kwargs = update.message.reply_text.call_args
    body = args[0]
    assert "Not sure" in body or "clarification" in body.lower() or "Try one of" in body
    # Must list the canonical commands so the user can self-correct.
    assert "/drill" in body and "/scan" in body and "/help" in body


@pytest.mark.asyncio
async def test_nl_fallback_dispatches_drill_with_ticker(monkeypatch):
    """High-confidence drill intent → routes to drill_command with
    context.args = [TICKER] uppercased."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    _stub_router_decision(
        monkeypatch, intent="drill", args={"ticker": "NVDA"}, confidence=0.92
    )

    captured: dict = {}

    async def _stub_drill(update, ctx):
        captured["args"] = list(ctx.args)

    monkeypatch.setattr(tg, "drill_command", _stub_drill)

    update = _make_update(chat_id=111, text="what's NVDA looking like")
    await tg.nl_fallback(update, SimpleNamespace())
    assert captured.get("args") == ["NVDA"]


@pytest.mark.asyncio
async def test_nl_fallback_drill_resolves_company_name_to_ticker(monkeypatch):
    """The router prompt teaches Haiku to translate company names
    (Constellation Energy → CEG). When that succeeds, NL dispatch must
    pass the resolved TICKER, not the company name, to drill_command."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    _stub_router_decision(
        monkeypatch, intent="drill", args={"ticker": "CEG"}, confidence=0.9
    )

    captured: dict = {}

    async def _stub_drill(update, ctx):
        captured["args"] = list(ctx.args)

    monkeypatch.setattr(tg, "drill_command", _stub_drill)

    update = _make_update(chat_id=111, text="run a drill on Constellation Energy")
    await tg.nl_fallback(update, SimpleNamespace())
    assert captured.get("args") == ["CEG"]


@pytest.mark.asyncio
async def test_nl_fallback_drill_with_thesis_arg(monkeypatch):
    """When the router extracts both ticker AND thesis, both must reach
    drill_command in the right positional order."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    _stub_router_decision(
        monkeypatch,
        intent="drill",
        args={"ticker": "AVGO", "thesis": "ai_cake"},
        confidence=0.95,
    )

    captured: dict = {}

    async def _stub_drill(update, ctx):
        captured["args"] = list(ctx.args)

    monkeypatch.setattr(tg, "drill_command", _stub_drill)

    update = _make_update(chat_id=111, text="drill AVGO on ai cake")
    await tg.nl_fallback(update, SimpleNamespace())
    assert captured.get("args") == ["AVGO", "ai_cake"]


@pytest.mark.asyncio
async def test_nl_fallback_dispatches_scan_with_no_args(monkeypatch):
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    _stub_router_decision(monkeypatch, intent="scan", confidence=0.85)

    captured: dict = {}

    async def _stub_scan(update, ctx):
        captured["called"] = True
        captured["args"] = list(ctx.args)

    monkeypatch.setattr(tg, "scan_command", _stub_scan)

    update = _make_update(chat_id=111, text="anything new today")
    await tg.nl_fallback(update, SimpleNamespace())
    assert captured.get("called") is True
    assert captured.get("args") == []


@pytest.mark.asyncio
async def test_nl_fallback_dispatches_status(monkeypatch):
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    _stub_router_decision(monkeypatch, intent="status", confidence=1.0)

    captured: dict = {}

    async def _stub(update, ctx):
        captured["called"] = True

    monkeypatch.setattr(tg, "status_command", _stub)
    update = _make_update(chat_id=111, text="status")
    await tg.nl_fallback(update, SimpleNamespace())
    assert captured.get("called") is True


@pytest.mark.asyncio
async def test_nl_fallback_dispatches_thesis(monkeypatch):
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    _stub_router_decision(
        monkeypatch, intent="thesis", args={"name": "ai_cake"}, confidence=0.9
    )

    captured: dict = {}

    async def _stub(update, ctx):
        captured["args"] = list(ctx.args)

    monkeypatch.setattr(tg, "thesis_command", _stub)
    update = _make_update(chat_id=111, text="remind me what's in ai cake")
    await tg.nl_fallback(update, SimpleNamespace())
    assert captured.get("args") == ["ai_cake"]


@pytest.mark.asyncio
async def test_nl_fallback_dispatches_note_with_ticker_and_text(monkeypatch):
    """/note expects positional args [TICKER, *text_words] — verify the
    router's `{ticker, text}` dict is correctly translated to that shape."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    _stub_router_decision(
        monkeypatch,
        intent="note",
        args={"ticker": "NVDA", "text": "trim 20% if Q3 misses"},
        confidence=0.85,
    )

    captured: dict = {}

    async def _stub(update, ctx):
        captured["args"] = list(ctx.args)

    monkeypatch.setattr(tg, "note_command", _stub)
    update = _make_update(chat_id=111, text="trim NVDA 20% if Q3 misses")
    await tg.nl_fallback(update, SimpleNamespace())
    # First arg must be ticker, remaining args reassemble to the note text.
    args = captured.get("args") or []
    assert args[0] == "NVDA"
    assert " ".join(args[1:]) == "trim 20% if Q3 misses"


@pytest.mark.asyncio
async def test_nl_fallback_note_missing_text_replies_with_clarification(monkeypatch):
    """Router gave us a note intent but no text arg — we must NOT pass an
    empty body to /note (which would create an empty Notion paragraph).
    Reply with a clarification instead."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    _stub_router_decision(
        monkeypatch, intent="note", args={"ticker": "NVDA"}, confidence=0.85
    )

    captured: dict = {"called": False}

    async def _stub(update, ctx):
        captured["called"] = True

    monkeypatch.setattr(tg, "note_command", _stub)
    update = _make_update(chat_id=111, text="note about NVDA")
    await tg.nl_fallback(update, SimpleNamespace())
    assert captured["called"] is False, "must NOT dispatch /note without text"
    # And the user must see a hint to retry properly.
    args, _ = update.message.reply_text.call_args
    assert "ticker" in args[0].lower() and "text" in args[0].lower()


@pytest.mark.asyncio
async def test_nl_fallback_analyze_explains_step_10e(monkeypatch):
    """/analyze isn't built yet — NL path must give the same honest UX
    as the inline keyboard's 'Synthesize custom' button."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    _stub_router_decision(
        monkeypatch,
        intent="analyze",
        args={"topic": "defense semis"},
        confidence=0.9,
    )

    update = _make_update(chat_id=111, text="analyze defense semis")
    await tg.nl_fallback(update, SimpleNamespace())
    args, _ = update.message.reply_text.call_args
    body = args[0]
    assert "10e" in body or "not built" in body.lower()
    # Tell the user the workable paths until /analyze ships.
    assert "/drill" in body and "general" in body


@pytest.mark.asyncio
async def test_nl_fallback_dispatches_help(monkeypatch):
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    _stub_router_decision(monkeypatch, intent="help", confidence=1.0)

    captured: dict = {}

    async def _stub(update, ctx):
        captured["called"] = True

    monkeypatch.setattr(tg, "help_command", _stub)
    update = _make_update(chat_id=111, text="help")
    await tg.nl_fallback(update, SimpleNamespace())
    assert captured.get("called") is True


@pytest.mark.asyncio
async def test_nl_fallback_dropped_for_unallowed(monkeypatch):
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    _stub_router_decision(monkeypatch, intent="drill", args={"ticker": "X"})

    captured: dict = {"called": False}

    async def _stub(update, ctx):
        captured["called"] = True

    monkeypatch.setattr(tg, "drill_command", _stub)
    update = _make_update(chat_id=999, text="drill X")
    await tg.nl_fallback(update, SimpleNamespace())
    assert captured["called"] is False
    update.message.reply_text.assert_not_called()


@pytest.mark.asyncio
async def test_nl_fallback_empty_text_is_a_noop(monkeypatch):
    """Empty / whitespace input must not waste an LLM classification call."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    called = {"n": 0}

    async def _classify(text):
        called["n"] += 1
        from utils.schemas import RouterDecision
        return RouterDecision(intent="unknown", args={}, confidence=0.0)

    from agents import router as r
    monkeypatch.setattr(r, "classify", _classify)

    update = _make_update(chat_id=111, text="   ")
    await tg.nl_fallback(update, SimpleNamespace())
    assert called["n"] == 0, "empty input must short-circuit before classify"
    update.message.reply_text.assert_not_called()


@pytest.mark.asyncio
async def test_drill_command_unknown_thesis_explains(monkeypatch):
    """User typed /drill NVDA fake → reject with 'thesis not found', show
    known slugs, suggest the corrected command."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    monkeypatch.setattr(tg, "_resolve_thesis_slug", lambda t, r: None)
    monkeypatch.setattr(tg, "_list_thesis_slugs", lambda: ["ai_cake", "construction"])
    update = _make_update(chat_id=111, text="/drill NVDA fake")
    context = SimpleNamespace(args=["NVDA", "fake"])
    await tg.drill_command(update, context)
    args, kwargs = update.message.reply_text.call_args
    body = args[0]
    assert "not found" in body.lower()
    assert "fake" in body
    assert "ai_cake" in body and "construction" in body
    assert kwargs.get("parse_mode") == "HTML"


@pytest.mark.asyncio
async def test_drill_command_ticker_in_no_universe_sends_choice_keyboard(monkeypatch):
    """User typed /drill AAPL (no thesis specified, ticker in no universe).
    Bot must reply with an inline keyboard (3 buttons) so the user
    explicitly picks generic vs analyze vs cancel — not silent fallback."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    # Force resolver to return None — simulates ticker-in-no-universe.
    monkeypatch.setattr(tg, "_resolve_thesis_slug", lambda t, r: None)
    monkeypatch.setattr(tg, "_list_thesis_slugs", lambda: ["ai_cake", "construction"])
    update = _make_update(chat_id=111, text="/drill AAPL")
    context = SimpleNamespace(args=["AAPL"])
    await tg.drill_command(update, context)
    args, kwargs = update.message.reply_text.call_args
    body = args[0]
    # Body must mention the ticker and be HTML-mode.
    assert "AAPL" in body
    assert "isn't in any" in body or "not in any" in body
    assert kwargs.get("parse_mode") == "HTML"
    # Inline keyboard with all three buttons must be attached.
    markup = kwargs.get("reply_markup")
    assert markup is not None, "expected an InlineKeyboardMarkup attached"
    button_data = [btn.callback_data for row in markup.inline_keyboard for btn in row]
    assert any(d.startswith("drill_generic:") for d in button_data)
    assert any(d.startswith("drill_analyze:") for d in button_data)
    assert any(d.startswith("drill_cancel:") for d in button_data)
    # Each callback must carry the ticker so the handler doesn't need to
    # remember per-chat state across messages.
    assert all(d.endswith(":AAPL") for d in button_data)
    # Button labels must show cost estimates per Q1 sign-off (Option A).
    button_labels = [btn.text for row in markup.inline_keyboard for btn in row]
    assert any("$0.50" in lbl or "0.50" in lbl for lbl in button_labels)
    assert any("$0.60" in lbl or "0.60" in lbl for lbl in button_labels)
    assert any(lbl == "Cancel" for lbl in button_labels)


@pytest.mark.asyncio
async def test_drill_command_dropped_for_unallowed(monkeypatch):
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update = _make_update(chat_id=999, text="/drill NVDA")
    context = SimpleNamespace(args=["NVDA"])
    await tg.drill_command(update, context)
    update.message.reply_text.assert_not_called()


# --- drill_choice_callback (inline-keyboard taps) -------------------------


def _make_callback_query(chat_id: int | None, data: str):
    """Minimal CallbackQuery-shaped mock. Has .data, .message.chat.id,
    .message.reply_text/edit_message_text, and .answer() (acks the tap)."""
    edit_mock = AsyncMock()
    answer_mock = AsyncMock()
    reply_mock = AsyncMock()
    chat = SimpleNamespace(id=chat_id) if chat_id is not None else None
    message = SimpleNamespace(
        chat=chat, reply_text=reply_mock, edit_message_text=edit_mock
    )
    query = SimpleNamespace(
        data=data,
        message=message,
        answer=answer_mock,
        edit_message_text=edit_mock,
    )
    update = SimpleNamespace(callback_query=query, effective_chat=chat)
    return update, edit_mock, answer_mock, reply_mock


@pytest.mark.asyncio
async def test_drill_choice_callback_dropped_for_unallowed(monkeypatch):
    """Tap from a non-allowlisted chat must not trigger any work — the
    decorator skips because callback_query has no .message in the
    @require_allowlist sense, so we enforce manually inside the handler."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update, edit, answer, _ = _make_callback_query(
        chat_id=999, data="drill_generic:AAPL"
    )
    await tg.drill_choice_callback(update, SimpleNamespace())
    edit.assert_not_called()


@pytest.mark.asyncio
async def test_drill_choice_callback_cancel_edits_message(monkeypatch):
    """Cancel button → edit the prompt message in place to show 'Cancelled'.
    No drill kicked off."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update, edit, answer, _ = _make_callback_query(
        chat_id=111, data="drill_cancel:AAPL"
    )
    await tg.drill_choice_callback(update, SimpleNamespace())
    answer.assert_called_once()  # spinner cleared
    edit.assert_called_once()
    args, kwargs = edit.call_args
    body = args[0] if args else kwargs.get("text", "")
    assert "Cancel" in body or "cancel" in body.lower()
    assert "AAPL" in body
    assert kwargs.get("parse_mode") == "HTML"


@pytest.mark.asyncio
async def test_drill_choice_callback_analyze_explains_step_10e(monkeypatch):
    """Analyze button → /analyze isn't built yet (Step 10e). Reply must
    say so honestly and point at workable paths (run /drill TICKER general
    or force a thematic thesis) instead of pretending to dispatch."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update, edit, answer, _ = _make_callback_query(
        chat_id=111, data="drill_analyze:AAPL"
    )
    await tg.drill_choice_callback(update, SimpleNamespace())
    answer.assert_called_once()
    edit.assert_called_once()
    args, kwargs = edit.call_args
    body = args[0] if args else kwargs.get("text", "")
    assert "10e" in body or "not built" in body.lower()
    assert "AAPL" in body
    # Must point at the workable alternative (drill with general)
    assert "general" in body
    assert kwargs.get("parse_mode") == "HTML"


@pytest.mark.asyncio
async def test_drill_choice_callback_generic_routes_to_runner(monkeypatch):
    """Generic button → edits the prompt to ack the choice, then calls
    `_run_drill_and_reply` with thesis_slug='general'. We mock the runner
    helper so the test doesn't actually drill — we just assert the dispatch."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update, edit, answer, _ = _make_callback_query(
        chat_id=111, data="drill_generic:AAPL"
    )

    captured: dict = {}

    async def _fake_run(reply_target, ticker, thesis_slug):
        captured["ticker"] = ticker
        captured["thesis_slug"] = thesis_slug

    monkeypatch.setattr(tg, "_run_drill_and_reply", _fake_run)
    await tg.drill_choice_callback(update, SimpleNamespace())
    answer.assert_called_once()
    edit.assert_called_once()
    # Must have routed through the shared runner helper with the general slug.
    assert captured["ticker"] == "AAPL"
    assert captured["thesis_slug"] == "general"


@pytest.mark.asyncio
async def test_drill_choice_callback_unknown_prefix_logged(monkeypatch):
    """Defensive: a malformed callback_data (different prefix) must not
    crash — log a warning and bail."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update, edit, answer, _ = _make_callback_query(
        chat_id=111, data="drill_made_up:AAPL"
    )
    # Should not raise.
    await tg.drill_choice_callback(update, SimpleNamespace())
    answer.assert_called_once()
    edit.assert_not_called()


# --- general thesis recognition -------------------------------------------


def test_resolve_thesis_slug_recognises_general_when_explicit():
    """User types `/drill AAPL general` — must use the Buffett-framework
    thesis even though AAPL isn't in any universe. Empty universe on the
    general thesis means it can't be auto-picked, but explicit force works."""
    assert tg._resolve_thesis_slug("AAPL", "general") == "general"


def test_general_thesis_loaded_with_buffett_thresholds():
    """Sanity: the general thesis on disk has Buffett-framework material
    thresholds — moats, leverage, accounting red flags. Pin a few so a
    future edit can't accidentally swap in a thematic set."""
    from utils.schemas import Thesis

    raw = json.loads(Path("theses/general.json").read_text())
    thesis = Thesis.model_validate(raw)
    signals = {t.signal for t in thesis.material_thresholds}
    # Buffett-classic moat / fortress signals must be present.
    assert "roe_ttm" in signals
    assert "roic_ttm" in signals
    assert "debt_to_equity" in signals
    assert "interest_coverage" in signals
    # Accounting red flags
    accounting_phrases = [
        t.value for t in thesis.material_thresholds
        if t.signal == "filing_mentions" and t.operator == "contains"
    ]
    assert "going concern" in accounting_phrases
    assert "material weakness" in accounting_phrases
    # Empty universe is intentional — it's a fall-through thesis.
    assert thesis.universe == []
    assert thesis.anchor_tickers == []


# --- _send_safe (retry + disk fallback) -----------------------------------


@pytest.mark.asyncio
async def test_send_safe_returns_true_on_first_attempt(monkeypatch):
    """Happy path — the message goes through immediately, no retries."""
    monkeypatch.setattr(tg, "_SEND_RETRY_DELAYS_S", (0, 0, 0))
    update = _make_update(chat_id=111, text="x")
    ok = await tg._send_safe(update, "hello", label="test")
    assert ok is True
    update.message.reply_text.assert_called_once()


@pytest.mark.asyncio
async def test_send_safe_retries_on_network_error(monkeypatch):
    """First two attempts hit NetworkError, third succeeds → returns True."""
    from telegram.error import NetworkError

    monkeypatch.setattr(tg, "_SEND_RETRY_DELAYS_S", (0, 0, 0))
    reply_mock = AsyncMock(
        side_effect=[NetworkError("boom"), NetworkError("boom"), None]
    )
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=111),
        message=SimpleNamespace(reply_text=reply_mock, text=""),
    )
    ok = await tg._send_safe(update, "hello", label="test")
    assert ok is True
    assert reply_mock.call_count == 3, "should have retried twice before success"


@pytest.mark.asyncio
async def test_send_safe_falls_back_to_disk_on_persistent_failure(
    monkeypatch, tmp_path
):
    """All retries fail → write the body to data_cache/telegram/pending/
    and return False so the caller knows delivery was lost."""
    from telegram.error import NetworkError

    monkeypatch.setattr(tg, "_SEND_RETRY_DELAYS_S", (0, 0, 0))
    monkeypatch.setattr(tg, "_SEND_FALLBACK_DIR", tmp_path / "pending")
    reply_mock = AsyncMock(side_effect=NetworkError("boom"))
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=111),
        message=SimpleNamespace(reply_text=reply_mock, text=""),
    )
    ok = await tg._send_safe(update, "expensive summary", label="drill-NVDA")
    assert ok is False
    assert reply_mock.call_count == len(tg._SEND_RETRY_DELAYS_S), (
        "should have used every retry slot"
    )
    # The body must end up on disk for recovery.
    files = list((tmp_path / "pending").glob("*drill-NVDA*"))
    assert len(files) == 1
    assert "expensive summary" in files[0].read_text()


@pytest.mark.asyncio
async def test_send_safe_does_not_retry_on_non_network_errors(monkeypatch):
    """A BadRequest from malformed content is a code bug, not a network
    blip — retrying just costs more failed sends. Bail immediately and
    write to disk."""

    monkeypatch.setattr(tg, "_SEND_RETRY_DELAYS_S", (0, 0, 0))
    reply_mock = AsyncMock(side_effect=ValueError("malformed"))
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=111),
        message=SimpleNamespace(reply_text=reply_mock, text=""),
    )
    ok = await tg._send_safe(update, "hello", label="test")
    assert ok is False
    assert reply_mock.call_count == 1, "non-network errors must NOT retry"


# --- _send_mc_chart -------------------------------------------------------


@pytest.mark.asyncio
async def test_send_mc_chart_sends_photo_with_caption(monkeypatch):
    """Happy path: state has DCF percentiles, samples are reproducible, the
    chart is rendered to PNG bytes, and reply_photo is called with the
    expected caption."""
    monkeypatch.setattr(tg, "_SEND_RETRY_DELAYS_S", (0, 0, 0))
    state = {
        "ticker": "NVDA",
        "thesis": {"name": "AI cake"},
        "monte_carlo": {
            "method": "dcf+multiple",
            "current_price": 205.66,
            "dcf": {"p10": 90.0, "p50": 195.0, "p90": 280.0},
        },
    }
    photo_mock = AsyncMock()
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=111),
        message=SimpleNamespace(reply_photo=photo_mock, text=""),
    )
    ok = await tg._send_mc_chart(update, state, label="test")
    assert ok is True
    photo_mock.assert_called_once()
    _, kwargs = photo_mock.call_args
    assert "caption" in kwargs
    assert "NVDA" in kwargs["caption"]
    assert "AI cake" in kwargs["caption"]
    # Caption shows the percentile bands so the user sees them inline.
    assert "$90" in kwargs["caption"] or "$91" in kwargs["caption"]
    assert "$195" in kwargs["caption"]
    assert kwargs.get("parse_mode") == "HTML"


@pytest.mark.asyncio
async def test_send_mc_chart_returns_false_when_mc_skipped(monkeypatch):
    """When MC was skipped (no DCF percentiles), there's nothing to
    render — the caller proceeds without sending a photo."""
    monkeypatch.setattr(tg, "_SEND_RETRY_DELAYS_S", (0, 0, 0))
    state = {"monte_carlo": {"method": "skipped"}}
    photo_mock = AsyncMock()
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=111),
        message=SimpleNamespace(reply_photo=photo_mock, text=""),
    )
    ok = await tg._send_mc_chart(update, state, label="test")
    assert ok is False
    photo_mock.assert_not_called()


@pytest.mark.asyncio
async def test_send_mc_chart_falls_back_to_disk_on_network_failure(
    monkeypatch, tmp_path
):
    """All retries fail → write the PNG to data_cache/telegram/pending/
    so the user can recover the chart from disk."""
    from telegram.error import NetworkError

    monkeypatch.setattr(tg, "_SEND_RETRY_DELAYS_S", (0, 0, 0))
    monkeypatch.setattr(tg, "_SEND_FALLBACK_DIR", tmp_path / "pending")
    state = {
        "ticker": "NVDA",
        "thesis": {"name": "AI cake"},
        "monte_carlo": {
            "method": "dcf+multiple",
            "current_price": 205.66,
            "dcf": {"p10": 90.0, "p50": 195.0, "p90": 280.0},
        },
    }
    photo_mock = AsyncMock(side_effect=NetworkError("boom"))
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=111),
        message=SimpleNamespace(reply_photo=photo_mock, text=""),
    )
    ok = await tg._send_mc_chart(update, state, label="drill-NVDA-chart")
    assert ok is False
    files = list((tmp_path / "pending").glob("*drill-NVDA-chart*.png"))
    assert len(files) == 1
    # PNG signature: first 8 bytes of any PNG
    assert files[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
