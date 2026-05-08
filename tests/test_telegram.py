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
    # Links — Streamlit is always rendered as a tappable <a href> so
    # macOS Telegram opens it on tap (iOS users get a "(Mac only)" hint
    # appended in the localhost branch).
    assert "ticker=NVDA" in out
    assert "thesis=ai_cake" in out
    assert "run_id=abc12345-deadbeef" in out
    assert "Open in Streamlit</a>" in out
    assert "(Mac only)" in out  # localhost suffix
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


def test_format_drill_summary_uses_127_not_localhost_in_url(monkeypatch):
    """Telegram silently strips `<a href>` tags pointing at `localhost`
    (verified by sending test messages and inspecting the API response's
    entities array — localhost links return zero entities). `127.0.0.1`
    is accepted untouched. Streamlit binds to both, so the rewrite is
    transparent on macOS. Without this, the user sees the link text but
    Telegram refuses to make it tappable.

    Pin the rewrite so a future edit can't reintroduce the localhost
    string and re-break the tap-to-open flow."""
    monkeypatch.delenv("STREAMLIT_PUBLIC_URL", raising=False)
    out = tg._format_drill_summary(_DEMO_STATE_FIXTURE, run_id="abc", thesis_slug="ai_cake")
    # The bare string `localhost` must NOT appear inside the rendered
    # Streamlit URL — Telegram would strip the link entity if it did.
    # Extract URLs from <a href="..."> and check.
    import re
    href_urls = re.findall(r'href="([^"]+)"', out)
    streamlit_urls = [u for u in href_urls if "ticker=" in u]
    assert streamlit_urls, "expected at least one Streamlit URL"
    for u in streamlit_urls:
        assert "localhost" not in u, (
            f"Streamlit URL contains 'localhost' which Telegram strips: {u!r}"
        )
        assert "127.0.0.1" in u, f"Streamlit URL must use 127.0.0.1: {u!r}"
    # And the (Mac only) hint still appears so iOS users know to switch.
    assert "(Mac only)" in out


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


def test_format_drill_summary_falls_back_to_127_streamlit(monkeypatch):
    """When STREAMLIT_PUBLIC_URL isn't set (Phase 0 — Step 12 sets it on
    the droplet), the link uses `127.0.0.1` rather than `localhost`.
    Telegram strips localhost-targeted `<a href>` tags server-side; the
    127 form survives. Streamlit binds to both, so the rewrite is
    transparent on macOS."""
    monkeypatch.delenv("STREAMLIT_PUBLIC_URL", raising=False)
    out = tg._format_drill_summary(_DEMO_STATE_FIXTURE, run_id="abc12345")
    assert "http://127.0.0.1:8501" in out
    assert "http://localhost:8501" not in out


def test_format_drill_summary_uses_public_url_when_set(monkeypatch):
    monkeypatch.setenv("STREAMLIT_PUBLIC_URL", "https://finaq.example.com")
    out = tg._format_drill_summary(_DEMO_STATE_FIXTURE, run_id="abc")
    assert "https://finaq.example.com/?ticker=NVDA" in out


def test_format_drill_summary_omits_notion_when_url_missing(monkeypatch):
    monkeypatch.delenv("STREAMLIT_PUBLIC_URL", raising=False)
    state = {**_DEMO_STATE_FIXTURE, "notion_report_url": ""}
    out = tg._format_drill_summary(state, run_id="abc", thesis_slug="ai_cake")
    assert "View in Notion" not in out
    # Streamlit link is always present as a tappable <a href> — macOS
    # Telegram opens localhost URLs fine when wrapped this way.
    assert "Open in Streamlit</a>" in out
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
    """An off-the-radar ticker (not in any thesis universe — including
    auto-generated adhoc_*) should return None rather than silently fall
    through to the first thesis. The drill_command surfaces /analyze as
    the right tool for ad-hoc topics. Use a fictional symbol so the test
    stays robust as the user runs more `/analyze` syntheses (which write
    `theses/adhoc_*.json` and could otherwise contain real tickers)."""
    assert tg._resolve_thesis_slug("ZZZZ_NOT_REAL", None) is None
    # Sanity check that the same ticker DOES resolve when explicitly forced
    assert tg._resolve_thesis_slug("ZZZZ_NOT_REAL", "ai_cake") == "ai_cake"


# --- Streamlit URL builder ------------------------------------------------


def test_build_streamlit_url_carries_query_params(monkeypatch):
    monkeypatch.setenv("STREAMLIT_PUBLIC_URL", "https://example.com")
    url = tg._build_streamlit_url("NVDA", "ai_cake", "deadbeef-1234")
    assert url == "https://example.com/?ticker=NVDA&thesis=ai_cake&run_id=deadbeef-1234"


def test_build_streamlit_url_omits_run_id_when_absent(monkeypatch):
    monkeypatch.setenv("STREAMLIT_PUBLIC_URL", "https://example.com/")  # trailing slash
    url = tg._build_streamlit_url("NVDA", "ai_cake", None)
    assert url == "https://example.com/?ticker=NVDA&thesis=ai_cake"


def test_build_streamlit_url_rewrites_localhost_to_127(monkeypatch):
    """Telegram strips `<a href>` tags pointing at `localhost` server-side
    (verified empirically). `127.0.0.1` survives. Streamlit binds to both,
    so we rewrite at URL-build time. Pin the rewrite so a future edit
    can't reintroduce the localhost string."""
    # Default fallback (no STREAMLIT_PUBLIC_URL set) — must rewrite.
    monkeypatch.delenv("STREAMLIT_PUBLIC_URL", raising=False)
    url = tg._build_streamlit_url("NVDA", "ai_cake", None)
    assert "127.0.0.1" in url
    assert "localhost" not in url
    # Explicit localhost in env var — also rewritten so the bot dev
    # workflow doesn't trip the same Telegram strip.
    monkeypatch.setenv("STREAMLIT_PUBLIC_URL", "http://localhost:8501")
    url = tg._build_streamlit_url("NVDA", "ai_cake", None)
    assert "127.0.0.1" in url
    assert "localhost" not in url
    # Public URL untouched — the rewrite only fires for localhost hosts.
    monkeypatch.setenv("STREAMLIT_PUBLIC_URL", "https://finaq.example.com")
    url = tg._build_streamlit_url("NVDA", "ai_cake", None)
    assert "finaq.example.com" in url
    assert "127.0.0.1" not in url


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


# --- /promote SLUG  +  /demote SLUG --------------------------------------


@pytest.mark.asyncio
async def test_promote_command_no_args_replies_with_usage(monkeypatch):
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update = _make_update(chat_id=111, text="/promote")
    context = SimpleNamespace(args=[])
    await tg.promote_command(update, context)
    args, _ = update.message.reply_text.call_args
    assert "Usage" in args[0] and "/promote" in args[0]


@pytest.mark.asyncio
async def test_promote_command_invokes_lifecycle(monkeypatch):
    """Happy path: bot calls `data.theses.promote_thesis` with the
    normalised `adhoc_*` slug and renders the success message."""
    from data import theses as _theses

    captured: dict = {}

    def _fake_promote(slug: str) -> tuple[bool, str]:
        captured["slug"] = slug
        return True, "promoted adhoc_defense_semis → defense_semis"

    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    monkeypatch.setattr(_theses, "promote_thesis", _fake_promote)

    update = _make_update(chat_id=111, text="/promote adhoc_defense_semis")
    context = SimpleNamespace(args=["adhoc_defense_semis"])
    await tg.promote_command(update, context)

    assert captured.get("slug") == "adhoc_defense_semis"
    args, _ = update.message.reply_text.call_args
    assert "Promoted" in args[0] or "✅" in args[0]


@pytest.mark.asyncio
async def test_promote_command_normalises_unprefixed_slug(monkeypatch):
    """Calling `/promote defense_semis` (no prefix) must auto-add `adhoc_`
    so the user can type either form."""
    from data import theses as _theses

    captured: dict = {}

    def _fake_promote(slug: str) -> tuple[bool, str]:
        captured["slug"] = slug
        return True, "ok"

    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    monkeypatch.setattr(_theses, "promote_thesis", _fake_promote)

    update = _make_update(chat_id=111, text="/promote defense_semis")
    context = SimpleNamespace(args=["defense_semis"])
    await tg.promote_command(update, context)
    assert captured.get("slug") == "adhoc_defense_semis"


@pytest.mark.asyncio
async def test_promote_command_surfaces_failure(monkeypatch):
    from data import theses as _theses

    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    monkeypatch.setattr(
        _theses, "promote_thesis",
        lambda slug: (False, "not found at theses/adhoc_x.json"),
    )

    update = _make_update(chat_id=111, text="/promote adhoc_x")
    context = SimpleNamespace(args=["adhoc_x"])
    await tg.promote_command(update, context)
    args, _ = update.message.reply_text.call_args
    assert "fail" in args[0].lower() or "⚠️" in args[0]
    assert "not found" in args[0]


@pytest.mark.asyncio
async def test_promote_command_dropped_for_unallowed(monkeypatch):
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update = _make_update(chat_id=999, text="/promote adhoc_x")
    context = SimpleNamespace(args=["adhoc_x"])
    await tg.promote_command(update, context)
    update.message.reply_text.assert_not_called()


@pytest.mark.asyncio
async def test_demote_command_no_args_replies_with_usage(monkeypatch):
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update = _make_update(chat_id=111, text="/demote")
    context = SimpleNamespace(args=[])
    await tg.demote_command(update, context)
    args, _ = update.message.reply_text.call_args
    assert "Usage" in args[0] and "/demote" in args[0]


@pytest.mark.asyncio
async def test_demote_command_invokes_lifecycle(monkeypatch):
    from data import theses as _theses

    captured: dict = {}

    def _fake_demote(slug: str) -> tuple[bool, str]:
        captured["slug"] = slug
        return True, "archived → 20260430_174812__nvda_halo.json"

    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    monkeypatch.setattr(_theses, "demote_thesis", _fake_demote)

    update = _make_update(chat_id=111, text="/demote nvda_halo")
    context = SimpleNamespace(args=["nvda_halo"])
    await tg.demote_command(update, context)

    assert captured.get("slug") == "nvda_halo"
    args, _ = update.message.reply_text.call_args
    assert "Demoted" in args[0] or "✅" in args[0]


@pytest.mark.asyncio
async def test_demote_command_surfaces_failure(monkeypatch):
    from data import theses as _theses

    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    monkeypatch.setattr(
        _theses, "demote_thesis",
        lambda slug: (False, "is an adhoc slug — use archive_thesis() instead"),
    )

    update = _make_update(chat_id=111, text="/demote adhoc_x")
    context = SimpleNamespace(args=["adhoc_x"])
    await tg.demote_command(update, context)
    args, _ = update.message.reply_text.call_args
    assert "fail" in args[0].lower() or "⚠️" in args[0]


@pytest.mark.asyncio
async def test_demote_command_dropped_for_unallowed(monkeypatch):
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update = _make_update(chat_id=999, text="/demote nvda_halo")
    context = SimpleNamespace(args=["nvda_halo"])
    await tg.demote_command(update, context)
    update.message.reply_text.assert_not_called()


def test_bot_commands_includes_promote_and_demote():
    """Both lifecycle commands must appear in BOT_COMMANDS so Telegram's
    /-autocomplete surfaces them."""
    names = {name for name, _ in tg.BOT_COMMANDS}
    assert "promote" in names
    assert "demote" in names


# --- /cio (Step 11.11) ---------------------------------------------------


@pytest.mark.asyncio
async def test_cio_command_no_args_runs_heartbeat(monkeypatch):
    """`/cio` with no args fires the heartbeat sweep."""
    from cio import cio as cio_orchestrator
    from cio import notify as cio_notify
    from cio.planner import CIODecision, Plan
    from data import state as _state

    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})

    fake_plan = Plan(
        decisions=[
            CIODecision(action="drill", ticker="NVDA", thesis="ai_cake",
                        rationale="x", confidence="high"),
        ],
        drill_budget=3,
    )
    captured: dict = {}

    async def _heartbeat(**kw):
        captured["mode"] = "heartbeat"
        return fake_plan, "summary text"

    async def _on_demand(*a, **kw):
        captured["mode"] = "on_demand"
        return fake_plan, "summary"

    monkeypatch.setattr(cio_orchestrator, "run_heartbeat", _heartbeat)
    monkeypatch.setattr(cio_orchestrator, "run_on_demand", _on_demand)
    monkeypatch.setattr(cio_notify, "write_to_notion_alert", lambda *a, **k: None)
    monkeypatch.setattr(_state, "recent_cio_runs", lambda *a, **k: [{"duration_s": 5.0}])

    update = _make_update(chat_id=111, text="/cio")
    context = SimpleNamespace(args=[])
    await tg.cio_command(update, context)

    assert captured.get("mode") == "heartbeat"
    # Two reply_text calls: ack + exec summary.
    assert update.message.reply_text.call_count == 2
    second_call = update.message.reply_text.call_args_list[1]
    assert "NVDA" in second_call.args[0]


@pytest.mark.asyncio
async def test_cio_command_ticker_runs_on_demand(monkeypatch):
    """`/cio NVDA` resolves to run_on_demand(NVDA, None)."""
    from cio import cio as cio_orchestrator
    from cio import notify as cio_notify
    from cio.planner import CIODecision, Plan
    from data import state as _state

    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})

    fake_plan = Plan(decisions=[
        CIODecision(action="reuse", ticker="NVDA", thesis="ai_cake",
                    rationale="still applies", confidence="medium",
                    reuse_run_id="r1"),
    ], drill_budget=3)
    captured: dict = {}

    async def _on_demand(ticker, thesis_slug=None, **kw):
        captured["ticker"] = ticker
        captured["thesis_slug"] = thesis_slug
        return fake_plan, "summary"

    monkeypatch.setattr(cio_orchestrator, "run_on_demand", _on_demand)
    monkeypatch.setattr(cio_notify, "write_to_notion_alert", lambda *a, **k: None)
    monkeypatch.setattr(_state, "recent_cio_runs", lambda *a, **k: [{"duration_s": 1.0}])

    update = _make_update(chat_id=111, text="/cio NVDA")
    context = SimpleNamespace(args=["NVDA"])
    await tg.cio_command(update, context)

    assert captured.get("ticker") == "NVDA"
    assert captured.get("thesis_slug") is None


@pytest.mark.asyncio
async def test_cio_command_ticker_thesis_pair(monkeypatch):
    """`/cio NVDA ai_cake` forces the explicit pair through to the orchestrator."""
    from cio import cio as cio_orchestrator
    from cio import notify as cio_notify
    from cio.planner import CIODecision, Plan
    from data import state as _state

    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})

    fake_plan = Plan(decisions=[
        CIODecision(action="dismiss", ticker="NVDA", thesis="ai_cake",
                    rationale="quiet", confidence="medium"),
    ], drill_budget=3)
    captured: dict = {}

    async def _on_demand(ticker, thesis_slug=None, **kw):
        captured["ticker"] = ticker
        captured["thesis_slug"] = thesis_slug
        return fake_plan, "summary"

    monkeypatch.setattr(cio_orchestrator, "run_on_demand", _on_demand)
    monkeypatch.setattr(cio_notify, "write_to_notion_alert", lambda *a, **k: None)
    monkeypatch.setattr(_state, "recent_cio_runs", lambda *a, **k: [{"duration_s": 1.0}])

    update = _make_update(chat_id=111, text="/cio NVDA ai_cake")
    context = SimpleNamespace(args=["NVDA", "ai_cake"])
    await tg.cio_command(update, context)

    assert captured.get("ticker") == "NVDA"
    assert captured.get("thesis_slug") == "ai_cake"


@pytest.mark.asyncio
async def test_cio_command_normalises_thesis_slug(monkeypatch):
    """User typing `/cio NVDA AI-Cake` should normalise the slug to
    `ai_cake` before dispatch (same convention every other command uses)."""
    from cio import cio as cio_orchestrator
    from cio import notify as cio_notify
    from cio.planner import CIODecision, Plan
    from data import state as _state

    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    fake_plan = Plan(decisions=[
        CIODecision(action="dismiss", ticker="NVDA", thesis="ai_cake",
                    rationale="x", confidence="low"),
    ], drill_budget=3)

    captured: dict = {}

    async def _on_demand(ticker, thesis_slug=None, **kw):
        captured["thesis_slug"] = thesis_slug
        return fake_plan, "summary"

    monkeypatch.setattr(cio_orchestrator, "run_on_demand", _on_demand)
    monkeypatch.setattr(cio_notify, "write_to_notion_alert", lambda *a, **k: None)
    monkeypatch.setattr(_state, "recent_cio_runs", lambda *a, **k: [{"duration_s": 1.0}])

    update = _make_update(chat_id=111, text="/cio NVDA AI-Cake")
    context = SimpleNamespace(args=["NVDA", "AI-Cake"])
    await tg.cio_command(update, context)
    assert captured.get("thesis_slug") == "ai_cake"


@pytest.mark.asyncio
async def test_cio_command_surfaces_orchestrator_failure(monkeypatch):
    """When the cycle raises, the bot replies with a graceful error
    instead of crashing the listener loop."""
    from cio import cio as cio_orchestrator

    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})

    async def _explode(**kw):
        raise RuntimeError("openrouter 503")

    monkeypatch.setattr(cio_orchestrator, "run_heartbeat", _explode)

    update = _make_update(chat_id=111, text="/cio")
    context = SimpleNamespace(args=[])
    await tg.cio_command(update, context)

    # Two reply_text calls: ack + error.
    assert update.message.reply_text.call_count == 2
    second_args, _ = update.message.reply_text.call_args_list[1]
    assert "CIO failed" in second_args[0]
    assert "openrouter 503" in second_args[0]


@pytest.mark.asyncio
async def test_cio_command_dropped_for_unallowed(monkeypatch):
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update = _make_update(chat_id=999, text="/cio")
    context = SimpleNamespace(args=[])
    await tg.cio_command(update, context)
    update.message.reply_text.assert_not_called()


def test_bot_commands_includes_cio():
    names = {name for name, _ in tg.BOT_COMMANDS}
    assert "cio" in names


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
    """The router prompt teaches the LLM to translate company names
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
async def test_nl_fallback_analyze_routes_to_analyze_command(monkeypatch):
    """NL `analyze defense semis` → routes to analyze_command with the
    topic as positional args. Step 10e replaced the 'coming soon' stub
    with the real synthesizer."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    _stub_router_decision(
        monkeypatch,
        intent="analyze",
        args={"topic": "defense semis"},
        confidence=0.9,
    )

    captured: dict = {}

    async def _stub(update, ctx):
        captured["args"] = list(ctx.args)

    monkeypatch.setattr(tg, "analyze_command", _stub)

    update = _make_update(chat_id=111, text="analyze defense semis")
    await tg.nl_fallback(update, SimpleNamespace())
    # Topic is split on whitespace and passed as positional args, the
    # same way Telegram's CommandHandler would split `/analyze defense semis`.
    assert captured.get("args") == ["defense", "semis"]


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
async def test_drill_choice_callback_analyze_acks_synthesis(monkeypatch):
    """Analyze button → bot acks the synthesis kicking off, then runs
    `_run_analyze` (which calls the synthesizer in TICKER mode). The
    detailed dispatch test
    `test_drill_choice_callback_analyze_runs_synthesizer` covers the
    end-to-end side; this one just pins that the user sees an ack edit."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})

    # Stub _run_analyze so the test doesn't actually call OpenRouter.
    async def _stub(update, *, topic, ticker):
        return None

    monkeypatch.setattr(tg, "_run_analyze", _stub)

    update, edit, answer, _ = _make_callback_query(
        chat_id=111, data="drill_analyze:AAPL"
    )
    await tg.drill_choice_callback(update, SimpleNamespace())
    answer.assert_called_once()
    edit.assert_called_once()
    args, kwargs = edit.call_args
    body = args[0] if args else kwargs.get("text", "")
    assert "AAPL" in body
    # The ack mentions synthesis is happening — no longer "not built".
    assert "Synthesizing" in body or "synthes" in body.lower()
    assert kwargs.get("parse_mode") == "HTML"


@pytest.mark.asyncio
async def test_drill_choice_callback_generic_routes_to_ingest_check(monkeypatch):
    """Generic button → edits the prompt to ack the choice, then routes
    through `_dispatch_drill_with_ingest_check` (which decides between
    silent drill and ingest-action keyboard based on whether the ticker
    is in ChromaDB). Step 10c.8b — closes the gap where the previous
    direct-to-runner dispatch silently produced inaccurate drills for
    not-yet-ingested tickers like FROG."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update, edit, answer, _ = _make_callback_query(
        chat_id=111, data="drill_generic:AAPL"
    )

    captured: dict = {}

    async def _fake_dispatch(reply_target, ticker, thesis_slug):
        captured["ticker"] = ticker
        captured["thesis_slug"] = thesis_slug

    monkeypatch.setattr(tg, "_dispatch_drill_with_ingest_check", _fake_dispatch)
    await tg.drill_choice_callback(update, SimpleNamespace())
    answer.assert_called_once()
    edit.assert_called_once()
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


# --- ambiguous-ticker keyboard --------------------------------------------


def test_list_matching_theses_returns_all_universes_excluding_general():
    """CEG is in BOTH ai_cake and nvda_halo — must return both. `general`
    is the catch-all (empty universe) and is excluded — it doesn't count
    as a thematic match for an ambiguous ticker."""
    matches = tg._list_matching_theses("CEG")
    assert "ai_cake" in matches
    assert "nvda_halo" in matches
    assert "general" not in matches


def test_list_matching_theses_returns_empty_for_non_universe_ticker():
    """Fictional ticker that's guaranteed not in any curated thesis OR
    auto-generated `adhoc_*` thesis. Step 10e introduced ad-hoc theses
    on disk (e.g. `theses/adhoc_aapl.json` after `/analyze AAPL`), so
    using AAPL/MSFT/etc. would be brittle as the user runs more
    syntheses."""
    matches = tg._list_matching_theses("ZZZZ_NOT_A_REAL_TICKER")
    assert matches == [], "fictional ticker shouldn't be in any thesis"


@pytest.mark.asyncio
async def test_drill_command_ambiguous_ticker_sends_thesis_choice_keyboard(monkeypatch):
    """User typed `/drill CEG` (no thesis) — CEG is in 2+ universes.
    Must send an inline keyboard letting the user pick the thematic
    framing rather than silently picking the first match."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    monkeypatch.setattr(
        tg, "_list_matching_theses", lambda t: ["ai_cake", "nvda_halo"]
    )
    update = _make_update(chat_id=111, text="/drill CEG")
    context = SimpleNamespace(args=["CEG"])
    await tg.drill_command(update, context)
    args, kwargs = update.message.reply_text.call_args
    body = args[0]
    assert "CEG" in body
    assert "2" in body  # mentions the count
    assert "ai_cake" in body and "nvda_halo" in body
    markup = kwargs.get("reply_markup")
    assert markup is not None
    button_data = [btn.callback_data for row in markup.inline_keyboard for btn in row]
    # Two thematic-pick buttons + general fallback + cancel.
    assert any(d == "drill_thesis:ai_cake:CEG" for d in button_data)
    assert any(d == "drill_thesis:nvda_halo:CEG" for d in button_data)
    assert any(d.startswith("drill_generic:") for d in button_data)
    assert any(d.startswith("drill_cancel:") for d in button_data)


@pytest.mark.asyncio
async def test_drill_command_single_match_dispatches_via_ingest_check(monkeypatch):
    """When a ticker matches exactly one thematic thesis, drill_command
    now routes through `_dispatch_drill_with_ingest_check` rather than
    silent dispatch. That helper decides between immediate drill (when
    ingested) and ingest-action keyboard (when not). This is the gap
    the user reported with /drill FROG (not in ChromaDB) silently
    producing an inaccurate report."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    monkeypatch.setattr(tg, "_list_matching_theses", lambda t: ["ai_cake"])

    captured: dict = {}

    async def _stub(reply_target, ticker, thesis_slug):
        captured["ticker"] = ticker
        captured["thesis_slug"] = thesis_slug

    monkeypatch.setattr(tg, "_dispatch_drill_with_ingest_check", _stub)
    update = _make_update(chat_id=111, text="/drill NVDA")
    context = SimpleNamespace(args=["NVDA"])
    await tg.drill_command(update, context)
    assert captured.get("thesis_slug") == "ai_cake"
    assert captured.get("ticker") == "NVDA"


@pytest.mark.asyncio
async def test_dispatch_drill_with_ingest_check_silent_when_ingested(monkeypatch):
    """If ChromaDB already has the ticker, no keyboard — straight to drill."""
    from data import chroma as _chroma
    from data import edgar as _edgar

    monkeypatch.setattr(_chroma, "has_ticker", lambda t: True)
    monkeypatch.setattr(_edgar, "has_filings_in_unsupported_kinds", lambda t: [])

    captured: dict = {}

    async def _stub_drill(reply_target, ticker, thesis_slug):
        captured["called"] = True
        captured["ticker"] = ticker
        captured["thesis_slug"] = thesis_slug

    monkeypatch.setattr(tg, "_run_drill_and_reply", _stub_drill)

    reply_mock = AsyncMock()
    reply_target = SimpleNamespace(reply_text=reply_mock)
    await tg._dispatch_drill_with_ingest_check(reply_target, "NVDA", "ai_cake")
    assert captured.get("called") is True
    assert captured.get("ticker") == "NVDA"
    # No keyboard message sent — straight dispatch.
    reply_mock.assert_not_called()


@pytest.mark.asyncio
async def test_dispatch_drill_with_ingest_check_warns_for_foreign_issuer(monkeypatch):
    """Foreign issuer (TSM, ASML) — ingestion won't help. Send a 1-line
    warning + drill anyway. No keyboard."""
    from data import chroma as _chroma
    from data import edgar as _edgar

    monkeypatch.setattr(_chroma, "has_ticker", lambda t: False)
    monkeypatch.setattr(_edgar, "has_filings_in_unsupported_kinds", lambda t: ["20-F"])

    drill_called = {"n": 0}

    async def _stub_drill(reply_target, ticker, thesis_slug):
        drill_called["n"] += 1

    monkeypatch.setattr(tg, "_run_drill_and_reply", _stub_drill)

    reply_mock = AsyncMock()
    reply_target = SimpleNamespace(reply_text=reply_mock)
    await tg._dispatch_drill_with_ingest_check(reply_target, "TSM", "ai_cake")
    assert drill_called["n"] == 1
    # The warning message must mention the unsupported kind so the user
    # knows why the drill will have empty filings.
    args, _ = reply_mock.call_args
    assert "20-F" in args[0]
    assert "foreign issuer" in args[0].lower()


@pytest.mark.asyncio
async def test_dispatch_drill_with_ingest_check_offers_keyboard_when_not_ingested(
    monkeypatch,
):
    """The user's reported case: ticker not in ChromaDB AND files
    supported kinds → must show the action keyboard before drilling so
    they can pick ingest-first."""
    from data import chroma as _chroma
    from data import edgar as _edgar

    monkeypatch.setattr(_chroma, "has_ticker", lambda t: False)
    monkeypatch.setattr(_edgar, "has_filings_in_unsupported_kinds", lambda t: [])

    drill_called = {"n": 0}

    async def _stub_drill(reply_target, ticker, thesis_slug):
        drill_called["n"] += 1

    monkeypatch.setattr(tg, "_run_drill_and_reply", _stub_drill)

    reply_mock = AsyncMock()
    reply_target = SimpleNamespace(reply_text=reply_mock)
    await tg._dispatch_drill_with_ingest_check(reply_target, "FROG", "adhoc_saas")
    # MUST NOT have drilled yet — waiting on user's keyboard tap.
    assert drill_called["n"] == 0
    # Must have sent a message with the action keyboard.
    reply_mock.assert_called_once()
    args, kwargs = reply_mock.call_args
    body = args[0]
    assert "FROG" in body
    assert "not ingested" in body.lower() or "ingest" in body.lower()
    markup = kwargs.get("reply_markup")
    assert markup is not None, "must attach an inline keyboard"
    # Keyboard must offer all three options: drill-now / ingest-then-drill / cancel.
    button_data = [
        btn.callback_data for row in markup.inline_keyboard for btn in row
    ]
    assert any(d.startswith("ad:adhoc_saas:FROG") for d in button_data)
    assert any(d.startswith("ai:adhoc_saas:FROG") for d in button_data)
    assert any(d.startswith("ax:adhoc_saas") for d in button_data)


@pytest.mark.asyncio
async def test_drill_command_explicit_thesis_also_runs_ingest_check(monkeypatch):
    """`/drill FROG adhoc_saas` (explicit thesis arg) must also route
    through the ingest check — the original bug was specifically about
    drills with adhoc theses where the ticker hasn't been ingested."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    monkeypatch.setattr(
        tg, "_resolve_thesis_slug",
        lambda t, r: r if r in ("adhoc_saas", "ai_cake") else None,
    )

    captured: dict = {}

    async def _stub(reply_target, ticker, thesis_slug):
        captured["ticker"] = ticker
        captured["thesis_slug"] = thesis_slug

    monkeypatch.setattr(tg, "_dispatch_drill_with_ingest_check", _stub)
    update = _make_update(chat_id=111, text="/drill FROG adhoc_saas")
    context = SimpleNamespace(args=["FROG", "adhoc_saas"])
    await tg.drill_command(update, context)
    assert captured.get("thesis_slug") == "adhoc_saas"
    assert captured.get("ticker") == "FROG"


@pytest.mark.asyncio
async def test_drill_thesis_callback_routes_to_picked_slug(monkeypatch):
    """Tap on a `drill_thesis:nvda_halo:CEG` button must dispatch with
    the picked slug. Step 10c.8b: now goes through the ingest-check
    helper rather than directly into `_run_drill_and_reply`."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update, edit, answer, _ = _make_callback_query(
        chat_id=111, data="drill_thesis:nvda_halo:CEG"
    )

    captured: dict = {}

    async def _stub(reply_target, ticker, thesis_slug):
        captured["ticker"] = ticker
        captured["thesis_slug"] = thesis_slug

    monkeypatch.setattr(tg, "_dispatch_drill_with_ingest_check", _stub)
    await tg.drill_choice_callback(update, SimpleNamespace())
    answer.assert_called_once()
    edit.assert_called_once()
    assert captured["ticker"] == "CEG"
    assert captured["thesis_slug"] == "nvda_halo"


# --- /analyze (Step 10e) --------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_command_no_args_replies_with_usage(monkeypatch):
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update = _make_update(chat_id=111, text="/analyze")
    context = SimpleNamespace(args=[])
    await tg.analyze_command(update, context)
    args, _ = update.message.reply_text.call_args
    assert "Usage" in args[0] and "/analyze" in args[0]


@pytest.mark.asyncio
async def test_analyze_command_too_short_topic_rejected(monkeypatch):
    """1-char topics are too vague — reject before paying $0.05."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update = _make_update(chat_id=111, text="/analyze x")
    context = SimpleNamespace(args=["x"])
    await tg.analyze_command(update, context)
    args, _ = update.message.reply_text.call_args
    assert "too short" in args[0].lower() or "tighter" in args[0].lower()


@pytest.mark.asyncio
async def test_analyze_command_dropped_for_unallowed(monkeypatch):
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update = _make_update(chat_id=999, text="/analyze defense semis")
    context = SimpleNamespace(args=["defense", "semis"])
    await tg.analyze_command(update, context)
    update.message.reply_text.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_sends_ticker_picker_keyboard_no_auto_drill(monkeypatch):
    """After Step 10c.7, /analyze must NOT auto-drill the top anchor.
    Instead it sends an inline keyboard with each anchor as a button so
    the user picks the ticker explicitly. This prevents the wasted
    drill-in on a ticker whose filings aren't yet in ChromaDB."""
    from agents import adhoc_thesis as at_mod
    from utils.schemas import Thesis

    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})

    fake_thesis = Thesis.model_validate({
        "name": "Defense semis (ad-hoc)",
        "summary": "Defense-exposed semis with multi-year backlog visibility.",
        "anchor_tickers": ["MRCY", "KTOS"],
        "universe": ["MRCY", "KTOS", "LMT", "RTX", "NOC"],
        "valuation": {
            "equity_risk_premium": 0.05,
            "erp_basis": "test",
            "terminal_growth_rate": 0.025,
            "terminal_growth_basis": "test",
            "discount_rate_floor": 0.07,
            "discount_rate_cap": 0.12,
        },
        "material_thresholds": [
            {"signal": "roe_ttm", "operator": "<", "value": 12, "unit": "percent"},
        ],
    })
    fake_result = at_mod.AdhocThesisResult(
        slug="adhoc_defense_semis",
        thesis=fake_thesis,
        path=Path("theses/adhoc_defense_semis.json"),
    )

    async def _stub_synth(**kwargs):
        return fake_result

    monkeypatch.setattr(at_mod, "synthesize_adhoc_thesis", _stub_synth)

    drill_called: dict = {"n": 0}

    async def _stub_drill(reply_target, ticker, thesis_slug):
        drill_called["n"] += 1

    monkeypatch.setattr(tg, "_run_drill_and_reply", _stub_drill)

    update = _make_update(chat_id=111, text="/analyze defense semis")
    context = SimpleNamespace(args=["defense", "semis"])
    await tg.analyze_command(update, context)

    # Drill MUST NOT have run (the user hasn't tapped a ticker yet).
    assert drill_called["n"] == 0

    # The last reply should carry the ticker-picker keyboard with both
    # anchors as buttons + Cancel.
    last_kwargs = update.message.reply_text.call_args_list[-1][1]
    markup = last_kwargs.get("reply_markup")
    assert markup is not None, "ticker picker keyboard missing"
    button_data = [
        btn.callback_data for row in markup.inline_keyboard for btn in row
    ]
    assert any(d == "ap:adhoc_defense_semis:MRCY" for d in button_data)
    assert any(d == "ap:adhoc_defense_semis:KTOS" for d in button_data)
    assert any(d.startswith("ax:") for d in button_data)
    button_labels = [
        btn.text for row in markup.inline_keyboard for btn in row
    ]
    # Anchors should be visually marked.
    assert any("⭐" in lbl and "MRCY" in lbl for lbl in button_labels)


@pytest.mark.asyncio
async def test_analyze_action_callback_pick_shows_action_keyboard(monkeypatch):
    """User taps a ticker on the picker → bot edits the prompt to
    show the action keyboard (drill now / ingest first / cancel)."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    # has_ticker → False (not ingested) so both options should appear.
    from data import chroma as _chroma
    from data import edgar as _edgar

    monkeypatch.setattr(_chroma, "has_ticker", lambda t: False)
    monkeypatch.setattr(_edgar, "has_filings_in_unsupported_kinds", lambda t: [])

    update, edit, answer, _ = _make_callback_query(
        chat_id=111, data="ap:adhoc_defense_semis:MRCY"
    )
    await tg.analyze_action_callback(update, SimpleNamespace())
    answer.assert_called_once()
    edit.assert_called_once()
    args, kwargs = edit.call_args
    body = args[0] if args else kwargs.get("text", "")
    markup = kwargs.get("reply_markup")
    assert "MRCY" in body
    assert "not ingested" in body.lower() or "ingest" in body.lower()
    button_data = [
        btn.callback_data for row in markup.inline_keyboard for btn in row
    ]
    assert any(d == "ad:adhoc_defense_semis:MRCY" for d in button_data)
    assert any(d == "ai:adhoc_defense_semis:MRCY" for d in button_data)
    assert any(d.startswith("ab:") for d in button_data)
    assert any(d.startswith("ax:") for d in button_data)


@pytest.mark.asyncio
async def test_analyze_action_callback_pick_omits_ingest_for_ingested_ticker(
    monkeypatch,
):
    """When the ticker is already ingested, the 'Ingest first' button
    must NOT appear — it's a no-op and would confuse."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    from data import chroma as _chroma
    from data import edgar as _edgar

    monkeypatch.setattr(_chroma, "has_ticker", lambda t: True)
    monkeypatch.setattr(_edgar, "has_filings_in_unsupported_kinds", lambda t: [])

    update, edit, answer, _ = _make_callback_query(
        chat_id=111, data="ap:adhoc_defense_semis:NVDA"
    )
    await tg.analyze_action_callback(update, SimpleNamespace())
    args, kwargs = edit.call_args
    markup = kwargs.get("reply_markup")
    button_data = [
        btn.callback_data for row in markup.inline_keyboard for btn in row
    ]
    assert any(d == "ad:adhoc_defense_semis:NVDA" for d in button_data)
    assert not any(
        d.startswith("ai:") for d in button_data
    ), "Ingest button must NOT appear when ticker is already ingested"


@pytest.mark.asyncio
async def test_analyze_action_callback_drill_now_dispatches(monkeypatch):
    """Tap on 'Drill now' → bot edits prompt + dispatches drill."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})

    captured: dict = {}

    async def _stub_drill(reply_target, ticker, thesis_slug):
        captured["ticker"] = ticker
        captured["thesis_slug"] = thesis_slug

    monkeypatch.setattr(tg, "_run_drill_and_reply", _stub_drill)

    update, edit, answer, _ = _make_callback_query(
        chat_id=111, data="ad:adhoc_defense_semis:MRCY"
    )
    await tg.analyze_action_callback(update, SimpleNamespace())
    edit.assert_called_once()
    assert captured.get("ticker") == "MRCY"
    assert captured.get("thesis_slug") == "adhoc_defense_semis"


@pytest.mark.asyncio
async def test_analyze_action_callback_ingest_then_drill(monkeypatch):
    """Tap on 'Ingest first, then drill' → bot ingests via
    `scripts.ingest_universe.ingest_ticker` then dispatches drill."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})

    ingest_called: dict = {}
    drill_called: dict = {}

    async def _stub_ingest(ticker):
        ingest_called["ticker"] = ticker
        return 42  # chunk count

    async def _stub_drill(reply_target, ticker, thesis_slug):
        drill_called["ticker"] = ticker
        drill_called["thesis_slug"] = thesis_slug

    import scripts.ingest_universe as iu

    monkeypatch.setattr(iu, "ingest_ticker", _stub_ingest)
    monkeypatch.setattr(tg, "_run_drill_and_reply", _stub_drill)

    update, edit, answer, _ = _make_callback_query(
        chat_id=111, data="ai:adhoc_defense_semis:MRCY"
    )
    await tg.analyze_action_callback(update, SimpleNamespace())
    # Both ingest + drill must have run, in order.
    assert ingest_called.get("ticker") == "MRCY"
    assert drill_called.get("ticker") == "MRCY"
    assert drill_called.get("thesis_slug") == "adhoc_defense_semis"
    # And the user saw an ingest-status reply (alongside any others).
    all_replies = [
        c.args[0] for c in update.callback_query.message.reply_text.call_args_list
    ]
    assert any("Ingested" in r or "chunks" in r.lower() for r in all_replies)


@pytest.mark.asyncio
async def test_analyze_action_callback_ingest_failure_falls_through_to_drill(
    monkeypatch,
):
    """Ingestion error must NOT block the drill — the user gets SOMETHING
    rather than a hung prompt. Failure is surfaced as a reply."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})

    async def _boom_ingest(ticker):
        raise RuntimeError("EDGAR rate limit")

    drill_called: dict = {}

    async def _stub_drill(reply_target, ticker, thesis_slug):
        drill_called["ticker"] = ticker

    import scripts.ingest_universe as iu

    monkeypatch.setattr(iu, "ingest_ticker", _boom_ingest)
    monkeypatch.setattr(tg, "_run_drill_and_reply", _stub_drill)

    update, edit, answer, _ = _make_callback_query(
        chat_id=111, data="ai:adhoc_defense_semis:MRCY"
    )
    await tg.analyze_action_callback(update, SimpleNamespace())
    # Drill ran despite ingest failure.
    assert drill_called.get("ticker") == "MRCY"
    # And the user saw the failure message.
    all_replies = [
        c.args[0] for c in update.callback_query.message.reply_text.call_args_list
    ]
    assert any("Ingestion failed" in r or "rate limit" in r for r in all_replies)


@pytest.mark.asyncio
async def test_analyze_action_callback_cancel_edits_message(monkeypatch):
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update, edit, answer, _ = _make_callback_query(
        chat_id=111, data="ax:adhoc_defense_semis"
    )
    await tg.analyze_action_callback(update, SimpleNamespace())
    edit.assert_called_once()
    args, kwargs = edit.call_args
    body = args[0] if args else kwargs.get("text", "")
    assert "Cancel" in body or "cancel" in body.lower()


@pytest.mark.asyncio
async def test_analyze_action_callback_back_re_renders_picker(
    monkeypatch, tmp_path
):
    """Tap on 'Back' → bot reloads the thesis from disk + re-renders
    the ticker-picker keyboard. Same shape as the initial picker."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    monkeypatch.setattr(tg, "THESES_DIR", tmp_path)
    # Write a real thesis JSON so _try_load_adhoc_thesis can reload.
    (tmp_path / "adhoc_defense_semis.json").write_text(
        json.dumps({
            "name": "Defense semis (ad-hoc)",
            "summary": "test summary",
            "anchor_tickers": ["MRCY", "KTOS"],
            "universe": ["MRCY", "KTOS", "LMT"],
            "valuation": {
                "equity_risk_premium": 0.05,
                "erp_basis": "x",
                "terminal_growth_rate": 0.025,
                "terminal_growth_basis": "x",
                "discount_rate_floor": 0.07,
                "discount_rate_cap": 0.12,
            },
            "material_thresholds": [
                {"signal": "roe_ttm", "operator": "<", "value": 12, "unit": "percent"}
            ],
        })
    )
    update, edit, answer, _ = _make_callback_query(
        chat_id=111, data="ab:adhoc_defense_semis"
    )
    await tg.analyze_action_callback(update, SimpleNamespace())
    edit.assert_called_once()
    args, kwargs = edit.call_args
    markup = kwargs.get("reply_markup")
    button_data = [
        btn.callback_data for row in markup.inline_keyboard for btn in row
    ]
    # Must re-show the ticker picker buttons.
    assert any(d == "ap:adhoc_defense_semis:MRCY" for d in button_data)
    assert any(d == "ap:adhoc_defense_semis:KTOS" for d in button_data)


@pytest.mark.asyncio
async def test_analyze_action_callback_dropped_for_unallowed(monkeypatch):
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update, edit, answer, _ = _make_callback_query(
        chat_id=999, data="ad:adhoc_defense_semis:MRCY"
    )
    await tg.analyze_action_callback(update, SimpleNamespace())
    edit.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_synth_failure_replies_gracefully(monkeypatch):
    """Synthesizer error (vague input, schema fail, LLM outage) → user
    sees a friendly message, no drill kicked off."""
    from agents import adhoc_thesis as at_mod

    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})

    fake_result = at_mod.AdhocThesisResult(
        slug="",
        thesis=None,  # type: ignore[arg-type]
        path=Path(),
        error="input too vague",
    )

    async def _stub_synth(**kwargs):
        return fake_result

    monkeypatch.setattr(at_mod, "synthesize_adhoc_thesis", _stub_synth)

    captured: dict = {"called": False}

    async def _stub_run(reply_target, ticker, thesis_slug):
        captured["called"] = True

    monkeypatch.setattr(tg, "_run_drill_and_reply", _stub_run)

    update = _make_update(chat_id=111, text="/analyze stocks")
    context = SimpleNamespace(args=["stocks"])
    await tg.analyze_command(update, context)
    assert captured["called"] is False, "must NOT drill on synth failure"
    args = [c.args[0] for c in update.message.reply_text.call_args_list]
    assert any("vague" in a.lower() for a in args)


@pytest.mark.asyncio
async def test_drill_choice_callback_analyze_runs_synthesizer(monkeypatch):
    """The "Synthesize custom" inline-keyboard branch routes to the
    synthesizer in TICKER mode. Step 10c.7 changed the post-synthesis
    flow: it no longer auto-drills; it now sends the ticker-picker
    keyboard. Pin the synthesizer dispatch + assert no auto-drill."""
    from agents import adhoc_thesis as at_mod
    from utils.schemas import Thesis

    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})

    fake_thesis = Thesis.model_validate({
        "name": "Apple (single-name ad-hoc)",
        "summary": "AAPL drill, single-name.",
        "anchor_tickers": ["AAPL"],
        "universe": ["AAPL", "MSFT", "GOOGL"],
        "valuation": {
            "equity_risk_premium": 0.05,
            "erp_basis": "test",
            "terminal_growth_rate": 0.025,
            "terminal_growth_basis": "test",
            "discount_rate_floor": 0.07,
            "discount_rate_cap": 0.11,
        },
        "material_thresholds": [
            {"signal": "roe_ttm", "operator": "<", "value": 12, "unit": "percent"},
        ],
    })
    fake_result = at_mod.AdhocThesisResult(
        slug="adhoc_aapl",
        thesis=fake_thesis,
        path=Path("theses/adhoc_aapl.json"),
    )

    captured: dict = {"synth_kwargs": None}

    async def _stub_synth(**kwargs):
        captured["synth_kwargs"] = kwargs
        return fake_result

    monkeypatch.setattr(at_mod, "synthesize_adhoc_thesis", _stub_synth)

    drill_called: dict = {"n": 0}

    async def _stub_run(reply_target, ticker, thesis_slug):
        drill_called["n"] += 1

    monkeypatch.setattr(tg, "_run_drill_and_reply", _stub_run)

    update, edit, answer, _ = _make_callback_query(
        chat_id=111, data="drill_analyze:AAPL"
    )
    await tg.drill_choice_callback(update, SimpleNamespace())

    # Synthesizer was called in TICKER mode (not topic mode).
    assert captured["synth_kwargs"] == {"topic": None, "ticker": "AAPL"}
    # And the drill MUST NOT have run yet — _run_analyze now ends at
    # the ticker-picker keyboard, not at a drill dispatch.
    assert drill_called["n"] == 0


# --- BOT_COMMANDS / set_my_commands (Step 10g) ----------------------------


def test_bot_commands_registry_covers_every_handler():
    """The /-autocomplete list must match the actual CommandHandler set
    registered in build_app. If we add a new command + forget to update
    BOT_COMMANDS, users won't see it in the autocomplete dropdown.

    Pin the symmetric set so a future edit is a one-stop change."""
    from telegram import BotCommand

    # Every entry must be a valid (name, description) pair within
    # Telegram's constraints.
    assert tg.BOT_COMMANDS, "registry empty — autocomplete won't work"
    for name, desc in tg.BOT_COMMANDS:
        assert 1 <= len(name) <= 32
        assert name.islower()
        assert all(c.isalnum() or c == "_" for c in name)
        assert 1 <= len(desc) <= 256

    # Names cover every real CommandHandler we wire up. /start is bound to
    # help_command but doesn't need to appear in the menu (Telegram
    # auto-shows it for first-contact). All other commands MUST be there.
    expected_names = {
        "drill", "analyze", "scan", "status", "note", "thesis", "theses", "help"
    }
    actual_names = {name for name, _ in tg.BOT_COMMANDS}
    missing = expected_names - actual_names
    assert not missing, f"BOT_COMMANDS missing: {sorted(missing)}"


@pytest.mark.asyncio
async def test_register_commands_calls_set_my_commands(monkeypatch):
    """`_register_commands` builds a list of telegram.BotCommand objects
    and ships it via `app.bot.set_my_commands(...)`."""
    from telegram import BotCommand

    captured: dict = {}

    set_my_commands_mock = AsyncMock()

    class _StubBot:
        async def set_my_commands(self, commands):
            captured["commands"] = list(commands)

    class _StubApp:
        bot = _StubBot()

    await tg._register_commands(_StubApp())
    sent = captured.get("commands") or []
    assert len(sent) == len(tg.BOT_COMMANDS)
    for cmd in sent:
        assert isinstance(cmd, BotCommand)
    sent_names = [c.command for c in sent]
    assert sent_names == [name for name, _ in tg.BOT_COMMANDS]


@pytest.mark.asyncio
async def test_register_commands_swallows_telegram_errors(monkeypatch):
    """A Telegram outage during set_my_commands must not crash bot startup
    — the bot still works without autocomplete (users type by hand)."""

    class _BoomBot:
        async def set_my_commands(self, commands):
            raise RuntimeError("telegram 502")

    class _StubApp:
        bot = _BoomBot()

    # Should not raise.
    await tg._register_commands(_StubApp())


@pytest.mark.asyncio
async def test_drill_thesis_callback_malformed_does_not_crash(monkeypatch):
    """A malformed `drill_thesis:` callback (missing pieces) must log
    + bail rather than crash the bot."""
    monkeypatch.setattr(tg, "_allowed_chat_ids", {111})
    update, edit, answer, _ = _make_callback_query(
        chat_id=111, data="drill_thesis:onlyone"
    )
    await tg.drill_choice_callback(update, SimpleNamespace())
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
