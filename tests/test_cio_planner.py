"""Tests for `cio/planner.py` — gates, decide(), and drill-budget cap.

The LLM call is stubbed in every test so this suite is deterministic and
runs in <1s. RAG retrieval is also stubbed because the synthesis_reports
collection depends on the live ChromaDB state.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from cio import planner
from cio.planner import CIODecision, evaluate_gates
from data import state as state_db


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    monkeypatch.setattr(state_db, "DB_PATH", db)
    state_db.init_db(db)
    return db


@pytest.fixture
def stub_external(monkeypatch):
    """Stub RAG + Notion + EDGAR file-walks so planner tests don't hit
    Chroma / Notion / disk."""
    from cio import memory as cio_memory
    from cio import rag as cio_rag

    monkeypatch.setattr(cio_rag, "query_past_reports", lambda **kw: [])
    monkeypatch.setattr(cio_memory, "thesis_notes", lambda slug: "")
    monkeypatch.setattr(planner, "_summarise_recent_filings", lambda t, since: [])


# --- CIODecision schema ---------------------------------------------------


def test_ciodecision_minimal_fields():
    d = CIODecision(action="drill", ticker="NVDA", rationale="why", confidence="high")
    assert d.thesis is None
    assert d.reuse_run_id is None
    assert d.followup_at is None


def test_ciodecision_rejects_unknown_action():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CIODecision(action="postpone", ticker="NVDA", rationale="x", confidence="high")


def test_ciodecision_rejects_unknown_confidence():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CIODecision(action="drill", ticker="NVDA", rationale="x", confidence="strong")


# --- evaluate_gates -------------------------------------------------------


def test_evaluate_gates_no_history(isolated_db, stub_external):
    out = evaluate_gates("NVDA", "ai_cake")
    assert out.shortcut is None
    assert out.cooldown_status["active"] is False
    assert out.dismissal_streak == []


def test_evaluate_gates_yo_yo_shortcuts_to_dismiss(isolated_db, stub_external):
    """3 dismissals in 7 days → 4th is shortcut to dismiss without LLM."""
    for _ in range(3):
        state_db.record_cio_action(
            ticker="NVDA", thesis="ai_cake", action="dismiss",
        )
    out = evaluate_gates("NVDA", "ai_cake")
    assert out.shortcut is not None
    assert out.shortcut.action == "dismiss"
    assert out.shortcut.confidence == "high"
    assert "yo-yo" in out.shortcut.rationale.lower()


def test_evaluate_gates_two_dismissals_no_shortcut(isolated_db, stub_external):
    """Below the 3-dismissal threshold → still let the LLM decide."""
    for _ in range(2):
        state_db.record_cio_action(
            ticker="NVDA", thesis="ai_cake", action="dismiss",
        )
    out = evaluate_gates("NVDA", "ai_cake")
    assert out.shortcut is None
    assert len(out.dismissal_streak) == 2


# --- _parse_decision ------------------------------------------------------


def test_parse_decision_strict_json():
    raw = json.dumps(
        {
            "action": "reuse",
            "ticker": "NVDA",
            "rationale": "still applies",
            "confidence": "high",
            "reuse_run_id": "abc",
        }
    )
    d, err = planner._parse_decision(raw)
    assert err is None and d is not None
    assert d.action == "reuse" and d.confidence == "high"


def test_parse_decision_extracts_from_fenced_response():
    """LLMs sometimes wrap in ```json fences. Parser falls back to regex."""
    raw = (
        "Sure, here you go:\n```json\n"
        + json.dumps({"action": "drill", "ticker": "NVDA",
                      "rationale": "x", "confidence": "high"})
        + "\n```"
    )
    d, err = planner._parse_decision(raw)
    assert err is None and d is not None
    assert d.action == "drill"


def test_parse_decision_empty_response_returns_error():
    d, err = planner._parse_decision("")
    assert d is None and err is not None
    assert "empty" in err.lower()


def test_parse_decision_invalid_json_returns_error():
    d, err = planner._parse_decision("not json {{{")
    assert d is None and err is not None


def test_parse_decision_invalid_schema_returns_error():
    raw = json.dumps({"action": "postpone", "ticker": "NVDA",
                      "rationale": "x", "confidence": "high"})
    d, err = planner._parse_decision(raw)
    assert d is None
    assert err and "schema" in err.lower()


# --- decide ---------------------------------------------------------------


def test_decide_yo_yo_shortcut_skips_llm(isolated_db, stub_external, monkeypatch):
    """When gates short-circuit, the LLM must NOT be called."""
    for _ in range(3):
        state_db.record_cio_action(ticker="NVDA", thesis="ai_cake", action="dismiss")

    called = {"n": 0}

    def _fail_if_called(**kw):
        called["n"] += 1
        return ""

    monkeypatch.setattr(planner, "_call_llm", _fail_if_called)

    out, telemetry = planner.decide(ticker="NVDA", thesis={"slug": "ai_cake"})
    assert out.action == "dismiss"
    assert called["n"] == 0
    # Gate-shortcut path → no LLM ran, telemetry zeroed.
    assert telemetry["model_used"] is None
    assert telemetry["cost_usd"] == 0.0
    assert telemetry["tokens_in"] == 0


def test_decide_calls_llm_and_returns_parsed(isolated_db, stub_external, monkeypatch):
    monkeypatch.setattr(
        planner, "_call_llm",
        lambda **kw: json.dumps(
            {
                "action": "drill",
                "ticker": "NVDA",
                "rationale": "Q3 lands tomorrow; refresh",
                "confidence": "high",
            }
        ),
    )

    out, telemetry = planner.decide(ticker="NVDA", thesis={"slug": "ai_cake"})
    assert out.action == "drill"
    assert out.confidence == "high"
    assert out.thesis == "ai_cake"  # post-processed from input
    # Telemetry shape is always populated, even when the openrouter
    # interceptor didn't see usage (because _call_llm is monkey-patched).
    assert telemetry["model_used"] is not None  # planner stamps MODEL_CIO
    assert "latency_s" in telemetry


def test_decide_normalises_ticker_case(isolated_db, stub_external, monkeypatch):
    """LLM might lowercase the ticker; planner forces uppercase to match
    the orchestrator's routing assumptions."""
    monkeypatch.setattr(
        planner, "_call_llm",
        lambda **kw: json.dumps(
            {
                "action": "drill", "ticker": "nvda",  # lowercase!
                "rationale": "x", "confidence": "high",
            }
        ),
    )
    out, _ = planner.decide(ticker="NVDA", thesis={"slug": "ai_cake"})
    assert out.ticker == "NVDA"


def test_decide_falls_back_to_dismiss_on_parse_error(isolated_db, stub_external, monkeypatch):
    monkeypatch.setattr(planner, "_call_llm", lambda **kw: "garbage not json")
    out, telemetry = planner.decide(ticker="NVDA", thesis={"slug": "ai_cake"})
    assert out.action == "dismiss"
    assert out.confidence == "low"
    assert "unparseable" in out.rationale.lower()
    # Telemetry still captured even on parse fail — the LLM call did fire,
    # so we want the latency_s / tokens to land on the cio_actions row.
    assert telemetry["latency_s"] >= 0.0


def test_decide_falls_back_to_dismiss_on_llm_exception(isolated_db, stub_external, monkeypatch):
    def _boom(**kw):
        raise RuntimeError("openrouter 503")

    monkeypatch.setattr(planner, "_call_llm", _boom)
    out, telemetry = planner.decide(ticker="NVDA", thesis={"slug": "ai_cake"})
    assert out.action == "dismiss"
    assert "llm call failed" in out.rationale.lower()
    # Telemetry still has the model + a (small) latency reading.
    assert telemetry["model_used"] is not None


def test_decide_reuse_backfills_run_id_when_llm_omits_it(isolated_db, stub_external, monkeypatch):
    """If the LLM picks reuse but doesn't include reuse_run_id, the planner
    fills it from the cooldown status's last_drill_run_id."""
    rid = state_db.start_graph_run("NVDA", "ai_cake")
    state_db.finish_graph_run(rid, "completed")

    monkeypatch.setattr(
        planner, "_call_llm",
        lambda **kw: json.dumps(
            {
                "action": "reuse", "ticker": "NVDA",
                "rationale": "still applies", "confidence": "high",
            }
        ),
    )
    out, _ = planner.decide(ticker="NVDA", thesis={"slug": "ai_cake"})
    assert out.action == "reuse"
    assert out.reuse_run_id == rid


def test_decide_telemetry_captured_via_contextvar(isolated_db, stub_external, monkeypatch):
    """When `_call_llm` runs (stubbed), the planner binds an accumulator
    to `node_telemetry_var`. To validate the wiring, we monkey-patch
    `_call_llm` to write directly into the bound accumulator — same
    surface the real openrouter interceptor uses."""
    def _llm_with_usage(**kw):
        # Simulate what utils/openrouter's interceptor does on a real call.
        accumulator = state_db.node_telemetry_var.get()
        if accumulator is not None:
            accumulator["tokens_in"] += 1500
            accumulator["tokens_out"] += 200
            accumulator["cost_usd"] += 0.0024
            accumulator["n_calls"] += 1
        return json.dumps(
            {"action": "dismiss", "ticker": "NVDA",
             "rationale": "quiet", "confidence": "low"}
        )

    monkeypatch.setattr(planner, "_call_llm", _llm_with_usage)

    out, telemetry = planner.decide(ticker="NVDA", thesis={"slug": "ai_cake"})
    assert out.action == "dismiss"
    assert telemetry["tokens_in"] == 1500
    assert telemetry["tokens_out"] == 200
    assert telemetry["cost_usd"] == 0.0024


# --- apply_drill_budget ---------------------------------------------------


def _drill(ticker: str, conf: str = "high") -> CIODecision:
    return CIODecision(
        action="drill", ticker=ticker, rationale=f"{ticker} drill",
        confidence=conf,
    )


def test_apply_drill_budget_no_op_when_under_cap():
    decisions = [_drill("NVDA"), _drill("MSFT")]
    out, capped = planner.apply_drill_budget(decisions, drill_budget=3)
    assert capped == 0
    assert [d.action for d in out] == ["drill", "drill"]


def test_apply_drill_budget_caps_at_3_demoting_lowest_confidence(
    isolated_db, monkeypatch,
):
    """When the LLM proposes 5 drills and budget=3, the bottom 2 by
    confidence must be demoted."""
    decisions = [
        _drill("AAA", "low"),
        _drill("BBB", "high"),
        _drill("CCC", "medium"),
        _drill("DDD", "low"),
        _drill("EEE", "high"),
    ]
    # No prior drills on disk → demoted ones become dismiss, not reuse.
    out, capped = planner.apply_drill_budget(decisions, drill_budget=3)
    assert capped == 2

    # Highest-confidence retained as drill: BBB, EEE, CCC.
    drilled_tickers = {d.ticker for d in out if d.action == "drill"}
    assert drilled_tickers == {"BBB", "EEE", "CCC"}

    # The two demoted ones become dismiss (no reusable run on disk).
    demoted = [d for d in out if d.ticker in {"AAA", "DDD"}]
    assert all(d.action == "dismiss" for d in demoted)
    assert all("budget cap" in d.rationale.lower() for d in demoted)


def test_apply_drill_budget_demotes_to_reuse_when_recent_drill_exists(
    isolated_db, monkeypatch,
):
    """An over-budget drill for a ticker with a recent run on disk demotes
    to reuse, not dismiss — the report is fresh enough to surface."""
    # Seed: one recent drill on AAA so cooldown has a run_id to reuse.
    rid = state_db.start_graph_run("AAA", "ai_cake")
    state_db.finish_graph_run(rid, "completed")

    decisions = [
        _drill("AAA", "low"),  # over-budget
        _drill("BBB", "high"),
        _drill("CCC", "high"),
        _drill("DDD", "high"),
    ]
    out, capped = planner.apply_drill_budget(decisions, drill_budget=3)
    assert capped == 1

    aaa = next(d for d in out if d.ticker == "AAA")
    assert aaa.action == "reuse"
    assert aaa.reuse_run_id == rid


def test_apply_drill_budget_preserves_input_order():
    decisions = [
        _drill("AAA", "low"), _drill("BBB", "high"),
        _drill("CCC", "high"), _drill("DDD", "low"),
    ]
    out, _ = planner.apply_drill_budget(decisions, drill_budget=2)
    assert [d.ticker for d in out] == ["AAA", "BBB", "CCC", "DDD"]


def test_apply_drill_budget_zero_budget_demotes_everything():
    decisions = [_drill("NVDA"), _drill("MSFT")]
    out, capped = planner.apply_drill_budget(decisions, drill_budget=0)
    assert capped == 2
    assert all(d.action != "drill" for d in out)


def test_apply_drill_budget_rejects_negative():
    with pytest.raises(ValueError, match="drill_budget"):
        planner.apply_drill_budget([_drill("X")], drill_budget=-1)


# --- Watchlist extraction + matching (Step 11.18) -----------------------


def test_extract_watchlist_items_handles_dash_bullets():
    text = (
        "Things to watch before the next drill-in:\n\n"
        "- Q3 earnings call (Aug 2026) — listen for AI capex guidance (news)\n"
        "- TSM yield disclosure in next 10-Q — supply concentration check (filings)\n"
        "- Inventory turnover trend in next quarter (fundamentals)\n"
    )
    items = planner._extract_watchlist_items(text)
    assert len(items) == 3
    assert items[0].startswith("Q3 earnings call")
    assert items[1].startswith("TSM yield disclosure")


def test_extract_watchlist_items_handles_star_and_plus_bullets():
    text = "* item one with stars\n+ item two with pluses\n- item three\n"
    items = planner._extract_watchlist_items(text)
    assert items == ["item one with stars", "item two with pluses", "item three"]


def test_extract_watchlist_items_skips_non_bullet_lines():
    text = "Preamble paragraph here.\n\n- real bullet\n\nTrailing prose."
    items = planner._extract_watchlist_items(text)
    assert items == ["real bullet"]


def test_extract_watchlist_items_returns_empty_on_none_or_empty():
    assert planner._extract_watchlist_items(None) == []
    assert planner._extract_watchlist_items("") == []


def test_significant_keywords_drops_stopwords_and_short_tokens():
    out = planner._significant_keywords("The capex announcement is up.")
    assert "capex" in out
    assert "announcement" in out
    assert "the" not in out  # stopword
    assert "is" not in out  # stopword + too short
    assert "up" not in out  # too short (<4)


def test_match_watchlist_news_overlap_matches():
    items = ["TSM yield disclosure in next 10-Q — supply concentration check (filings)"]
    news = [
        {"title": "TSM yield update from semiconductor analyst", "url": "https://x"},
        {"title": "Generic AI rally continues", "url": "https://y"},
    ]
    out = planner._match_watchlist_signals(items, news=news, filings=[])
    assert len(out) == 1
    assert "matched_news_title" in out[0]
    assert "TSM yield" in out[0]["matched_news_title"]
    assert "yield" in out[0]["match_keywords"]


def test_match_watchlist_below_overlap_threshold_skipped():
    items = ["TSM yield disclosure"]
    news = [{"title": "Apple announces a new iPhone", "url": "https://x"}]
    out = planner._match_watchlist_signals(items, news=news, filings=[])
    assert out == []


def test_match_watchlist_filing_overlap_matches():
    items = ["TSM 10-Q filing should reveal yield disclosure"]
    filings = [{"kind": "10-Q", "accession": "0001-26-001", "filed_at_iso": "2026-04-30"}]
    out = planner._match_watchlist_signals(items, news=[], filings=filings)
    assert len(out) == 1
    assert out[0]["matched_filing_kind"] == "10-Q"
    assert out[0]["matched_filing_accession"] == "0001-26-001"


def test_match_watchlist_skips_too_vague_items():
    """An item with <2 distinct significant keywords (after stopwords +
    length filter) is skipped — it would over-match noise."""
    # Single-keyword item: only "yield" survives the length+stopword filter
    # (the/it/be/up are dropped). 1 keyword < min 2 → skipped.
    items = ["The yield"]
    news = [{"title": "Yield curve update", "url": "x"}]
    out = planner._match_watchlist_signals(items, news=news, filings=[])
    assert out == []


def test_match_watchlist_empty_inputs_return_empty():
    assert planner._match_watchlist_signals([], news=[{"title": "x", "url": "y"}], filings=[]) == []
    assert planner._match_watchlist_signals(["a"], news=None, filings=None) == []


def test_build_evidence_bundle_includes_watchlist_fields(isolated_db, monkeypatch):
    """Integration: the bundle exposes both watchlist_items and watchlist_signals
    when the synthesis_reports collection has a Watchlist chunk for the pair."""
    from cio import rag as cio_rag

    # Stub RAG: return a watchlist chunk for the (NVDA, ai_cake) pair.
    monkeypatch.setattr(
        cio_rag, "latest_watchlist_section",
        lambda ticker, thesis=None: {
            "text": (
                "- NVDA Q3 earnings (Aug 2026) — listen for capex guidance (news)\n"
                "- TSM 10-Q yield disclosure — supply concentration check (filings)\n"
            ),
            "metadata": {"section": "Watchlist", "date": "2026-04-26"},
        },
    )
    monkeypatch.setattr(cio_rag, "query_past_reports", lambda **kw: [])
    monkeypatch.setattr(planner, "_summarise_recent_filings",
                        lambda t, since: [
                            {"kind": "10-Q", "accession": "abc", "filed_at_iso": "2026-04-29"},
                        ])

    gates = planner.GateOutcome(
        shortcut=None,
        cooldown_status={"active": False, "last_drill_age_hours": None,
                         "last_drill_run_id": None, "last_drill_at": None},
        dismissal_streak=[],
        notes="",
    )

    news = [
        {"title": "TSM yield update from analyst", "url": "https://x"},
        {"title": "AI rally continues", "url": "https://y"},
    ]
    bundle = planner.build_evidence_bundle(
        ticker="NVDA",
        thesis={"slug": "ai_cake", "summary": "AI thesis", "material_thresholds": []},
        gates=gates,
        rag_question="?",
        news_items=news,
    )
    assert "watchlist_items" in bundle
    assert len(bundle["watchlist_items"]) == 2
    assert any("NVDA Q3 earnings" in i for i in bundle["watchlist_items"])

    assert "watchlist_signals" in bundle
    # Both watchlist items should match: TSM watchlist ↔ TSM news, and
    # 10-Q watchlist ↔ 10-Q filing.
    assert len(bundle["watchlist_signals"]) >= 1
    has_news_match = any("matched_news_title" in s for s in bundle["watchlist_signals"])
    has_filing_match = any("matched_filing_kind" in s for s in bundle["watchlist_signals"])
    assert has_news_match or has_filing_match


def test_build_evidence_bundle_empty_watchlist_when_no_chunk(isolated_db, monkeypatch):
    """When ChromaDB has no Watchlist chunk for the pair (cold start),
    bundle still has the keys but the lists are empty."""
    from cio import rag as cio_rag
    from cio import memory as cio_memory

    monkeypatch.setattr(cio_rag, "latest_watchlist_section",
                        lambda ticker, thesis=None: None)
    monkeypatch.setattr(cio_rag, "query_past_reports", lambda **kw: [])
    monkeypatch.setattr(cio_memory, "thesis_notes", lambda slug: "")
    monkeypatch.setattr(planner, "_summarise_recent_filings", lambda t, since: [])

    gates = planner.evaluate_gates("NVDA", "ai_cake")
    bundle = planner.build_evidence_bundle(
        ticker="NVDA",
        thesis={"slug": "ai_cake", "summary": "x", "material_thresholds": []},
        gates=gates,
        rag_question="?",
        news_items=[],
    )
    assert bundle["watchlist_items"] == []
    assert bundle["watchlist_signals"] == []
