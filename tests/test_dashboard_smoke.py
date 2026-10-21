"""End-to-end Streamlit dashboard smoke tests.

`streamlit.testing.v1.AppTest` runs a Streamlit script in-process and
exposes its widgets / errors / output as Python objects. Way more reliable
than booting Streamlit and probing with curl — curl only confirms the
server is up; AppTest actually executes the rendering code path and
surfaces any Python exception.

These tests catch:
  - Import errors (sys.path bootstrap not working).
  - Path-relative bugs (e.g. `page_link("ui/app.py")` when entrypoint is
    `ui/app.py` and Streamlit expects paths relative to its dir).
  - Component rendering failures (e.g. `st.dataframe` on a malformed dict).
  - Sidebar / state-init crashes.
  - **Streamlit log-warnings** (e.g. the session-state-API antipattern
    where a widget has both `value=` and `session_state[key]` set —
    these are logged but don't raise, so AppTest's `at.exception` misses
    them. We catch them via `caplog`.

Each test loads ONE page via AppTest, runs it once, asserts no exception.
We do NOT exercise interaction (button clicks, form submits) here — that
belongs in higher-level integration tests if/when they're added.

Why this isn't already in conftest auto-fixtures: AppTest is heavy
(~0.5s per page), so we run these explicitly under the dashboard-smoke
test file and not on every test in the suite.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

# Streamlit warnings we want to fail the test on. These are signals of a
# real bug that won't show up in `at.exception` — e.g. the dual
# `value=`+`session_state[key]` antipattern only logs, doesn't raise.
# We deliberately do NOT include "missing ScriptRunContext" here — that's
# fired on every out-of-runtime import of a module that uses st.cache_*,
# which is unavoidable when AppTest loads a page.
_FORBIDDEN_WARNING_PHRASES = (
    "Session State API",  # widget has both value= and session_state[key] set
    "StreamlitPageNotFoundError",  # st.page_link path mismatch
    "DuplicateWidgetID",  # two widgets with the same key
)


def _streamlit_warning_records(records) -> list[logging.LogRecord]:
    """Filter a record list to ones that look like real Streamlit-level
    warnings worth failing on."""
    out: list[logging.LogRecord] = []
    for r in records:
        if r.levelno < logging.WARNING:
            continue
        msg = r.getMessage()
        if any(phrase in msg for phrase in _FORBIDDEN_WARNING_PHRASES):
            out.append(r)
    return out


class _StreamlitWarningCapture:
    """Captures warnings from the entire `streamlit.*` logger tree.

    pytest's `caplog` only sees records that propagate to the root logger;
    Streamlit's `streamlit.logger.get_logger` sets `propagate=False` on
    every logger it creates (see streamlit/logger.py:124). Attaching a
    handler to the parent `streamlit` logger therefore catches NOTHING —
    the records never bubble up.

    Workaround: walk every existing Streamlit logger AND monkey-patch
    `streamlit.logger.get_logger` so it auto-attaches our handler to any
    logger created during the test (Streamlit creates them lazily). At
    teardown we remove handlers and restore the patched function.
    """

    def __init__(self) -> None:
        self.records: list[logging.LogRecord] = []
        self._handler: logging.Handler | None = None
        self._patched_loggers: list[logging.Logger] = []
        self._original_get_logger = None

    def _make_handler(self) -> logging.Handler:
        sl = self

        class _Handler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                sl.records.append(record)

        return _Handler(level=logging.WARNING)

    def _attach_to(self, logger_name: str) -> None:
        if self._handler is None:
            return
        lg = logging.getLogger(logger_name)
        if self._handler not in lg.handlers:
            lg.addHandler(self._handler)
            self._patched_loggers.append(lg)

    def __enter__(self) -> _StreamlitWarningCapture:
        self._handler = self._make_handler()

        # 1. Attach to every already-existing streamlit.* logger.
        for name in list(logging.Logger.manager.loggerDict.keys()):
            if name == "streamlit" or name.startswith("streamlit."):
                self._attach_to(name)

        # 2. Patch streamlit.logger.get_logger so any future logger also
        #    gets the handler. Streamlit creates child loggers lazily on
        #    first use, so this catches the policies / runtime / etc.
        try:
            from streamlit import logger as _st_logger

            self._original_get_logger = _st_logger.get_logger

            def _wrapped_get_logger(name: str):
                lg = self._original_get_logger(name)
                if self._handler is not None and self._handler not in lg.handlers:
                    lg.addHandler(self._handler)
                    self._patched_loggers.append(lg)
                return lg

            _st_logger.get_logger = _wrapped_get_logger
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        import contextlib

        if self._handler is not None:
            for lg in self._patched_loggers:
                with contextlib.suppress(ValueError):
                    lg.removeHandler(self._handler)
        if self._original_get_logger is not None:
            try:
                from streamlit import logger as _st_logger

                _st_logger.get_logger = self._original_get_logger
            except Exception:
                pass
        self._handler = None
        self._patched_loggers.clear()
        self._original_get_logger = None

PROJECT_ROOT = Path(__file__).parents[1]
APP_DIR = PROJECT_ROOT / "ui"
PAGES_DIR = APP_DIR / "pages"

# Streamlit AppTest needs the entrypoint script's path. For pages, the
# library historically required them to be exercised through the
# multi-page-app harness; for our smoke tests it's simpler to point
# AppTest at each page's file directly. Both approaches work because
# every page ends with `main()` and runs unconditionally.

DASHBOARD_TIMEOUT_S = 30


@pytest.fixture
def _no_query_params(monkeypatch):
    """The main app reads query params (?ticker=NVDA&thesis=...) on first
    render — AppTest defaults to no params, which is fine; this fixture is
    a placeholder if we want to inject params later."""
    return None


# --- Main entrypoint --------------------------------------------------------


def test_main_dashboard_renders_without_exception(_no_query_params):
    """Loads ui/app.py end to end. Catches the page_link path bug and any
    other startup issue. Also asserts no Streamlit-level warnings fire
    (e.g. the dual `value=`+`session_state[key]` antipattern, which logs
    a warning but does NOT raise)."""
    with _StreamlitWarningCapture() as cap:
        at = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=DASHBOARD_TIMEOUT_S)
        at.run()
    if at.exception:
        excs = at.exception
        msgs = "\n".join(
            f"  - {e.message}\n{(e.stack_trace or '') if hasattr(e, 'stack_trace') else ''}"
            for e in excs
        )
        pytest.fail(f"Streamlit raised {len(excs)} exception(s):\n{msgs}")
    bad = _streamlit_warning_records(cap.records)
    if bad:
        pytest.fail(
            "Streamlit emitted forbidden warnings during dashboard render:\n"
            + "\n".join(f"  - [{r.name}] {r.getMessage()[:200]}" for r in bad)
        )


# --- Sub-pages (auto-discovered by Streamlit, but testable directly) -------


@pytest.mark.parametrize(
    "page_file",
    [
        "architecture.py",
        "methodology.py",
        "mission_control.py",
        "direct_agent.py",
        "new_thesis.py",
        "theses_admin.py",
        "backtest.py",
        "halo_graph.py",
    ],
)
def test_each_page_renders_without_exception(page_file):
    """Every dashboard page must import + render without raising AND without
    emitting any forbidden Streamlit warning."""
    with _StreamlitWarningCapture() as cap:
        at = AppTest.from_file(
            str(PAGES_DIR / page_file), default_timeout=DASHBOARD_TIMEOUT_S
        )
        at.run()
    if at.exception:
        excs = at.exception
        msgs = "\n".join(
            f"  - {e.message}\n{(e.stack_trace or '') if hasattr(e, 'stack_trace') else ''}"
            for e in excs
        )
        pytest.fail(f"Page {page_file} raised {len(excs)} exception(s):\n{msgs}")
    bad = _streamlit_warning_records(cap.records)
    if bad:
        pytest.fail(
            f"Page {page_file} emitted forbidden warnings:\n"
            + "\n".join(f"  - [{r.name}] {r.getMessage()[:200]}" for r in bad)
        )


# --- Sidebar contents -------------------------------------------------------


def test_main_dashboard_sidebar_has_thesis_dropdown_and_ticker_input():
    """Exercises the sidebar widgets actually appear. If page_link or any
    sidebar render path crashes, this test catches it."""
    at = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=DASHBOARD_TIMEOUT_S)
    at.run()
    assert not at.exception

    # Thesis selectbox should be present
    selectboxes = at.selectbox
    assert any(
        sb.label == "Thesis" for sb in selectboxes
    ), f"Thesis dropdown missing; saw: {[s.label for s in selectboxes]}"

    # Ticker text input should be present
    text_inputs = at.text_input
    assert any(
        ti.label == "Ticker" for ti in text_inputs
    ), f"Ticker text input missing; saw: {[t.label for t in text_inputs]}"


# --- Architecture page renders agent cards + topology ----------------------


def test_architecture_page_renders_agent_cards_and_topology():
    at = AppTest.from_file(
        str(PAGES_DIR / "architecture.py"), default_timeout=DASHBOARD_TIMEOUT_S
    )
    at.run()
    assert not at.exception
    # Should have agent-card sections — look for the agent names in the rendered markdown
    markdown_blocks = " ".join(m.value for m in at.markdown)
    for name in ("Fundamentals", "Filings", "News", "Risk", "Monte Carlo", "Synthesis"):
        assert name in markdown_blocks, f"Architecture page missing '{name}' agent card"


# --- Methodology page renders thesis selector ------------------------------


def test_methodology_page_offers_thesis_selector():
    at = AppTest.from_file(
        str(PAGES_DIR / "methodology.py"), default_timeout=DASHBOARD_TIMEOUT_S
    )
    at.run()
    assert not at.exception
    # Methodology has a thesis dropdown
    selectboxes = at.selectbox
    assert any(
        sb.label == "Thesis" for sb in selectboxes
    ), f"Methodology missing thesis dropdown; saw: {[s.label for s in selectboxes]}"


# --- Mission Control reads state.db without crashing on empty -------------


def test_mission_control_renders_with_empty_state_db():
    """When state.db is freshly empty (autouse fixture redirects it to
    tmp), Mission Control should display 'No graph runs recorded yet'
    rather than crash."""
    at = AppTest.from_file(
        str(PAGES_DIR / "mission_control.py"), default_timeout=DASHBOARD_TIMEOUT_S
    )
    at.run()
    assert not at.exception


def test_mission_control_renders_with_populated_state_db():
    """Pre-populate state.db with one run, then render Mission Control.
    Confirms the recent-runs table + chart paths actually exercise."""
    from data import state as state_db

    run_id = state_db.start_graph_run("NVDA", "ai_cake")
    state_db.record_node_run(
        run_id,
        "fundamentals",
        state_db._now_iso(),
        state_db._now_iso(),
        2.4,
        "completed",
    )
    state_db.finish_graph_run(run_id, "completed", confidence="medium", duration_s=12.5)

    at = AppTest.from_file(
        str(PAGES_DIR / "mission_control.py"), default_timeout=DASHBOARD_TIMEOUT_S
    )
    at.run()
    assert not at.exception
    # KPI cards are custom HTML (spend sparkline, 7d run breakdown) rather
    # than st.metric — assert the rendered markdown carries them.
    blob = " ".join(m.value for m in at.markdown)
    assert "Spend today" in blob
    assert "Runs · 7d" in blob


# --- Direct Agent page selectors ------------------------------------------


def test_direct_agent_load_demo_finds_run_id_suffixed_files(tmp_path, monkeypatch):
    """Regression: `_try_load_demo` previously hard-coded the legacy
    `{TICKER}__{slug}.json` filename, ignoring the run_id-suffixed files
    the runner started saving in task #47. The Direct Agent UI then fell
    back to a minimal state and `ask()` returned 'state.X is empty' for
    every agent — the user reported "no agents are responding".

    This test simulates the runner's output format and asserts the loader
    finds the most recent file."""
    import importlib

    da = importlib.import_module("ui.pages.direct_agent")
    monkeypatch.setattr(da, "DEMO_DIR", tmp_path)

    # Three runs of the same (NVDA, ai_cake) — most recent should win.
    import time

    (tmp_path / "NVDA__ai_cake__deadbeef.json").write_text('{"ticker":"NVDA","run_id":"old"}')
    time.sleep(0.05)  # ensure mtime ordering
    (tmp_path / "NVDA__ai_cake__cafef00d.json").write_text('{"ticker":"NVDA","run_id":"newer"}')
    time.sleep(0.05)
    (tmp_path / "NVDA__ai_cake__feedface.json").write_text('{"ticker":"NVDA","run_id":"newest"}')

    state = da._try_load_demo("NVDA", "ai_cake")
    assert state is not None, "loader must find run_id-suffixed files"
    assert state["run_id"] == "newest", "loader must pick the most recent file"


def test_direct_agent_load_demo_falls_back_to_legacy_filename(tmp_path, monkeypatch):
    """A subset of demos predate the run_id suffix (e.g. shipped
    `data_cache/demos/NVDA__ai_cake.json` from the demo seed). The loader
    must still find these so cached demos keep working."""
    import importlib

    da = importlib.import_module("ui.pages.direct_agent")
    monkeypatch.setattr(da, "DEMO_DIR", tmp_path)
    (tmp_path / "NVDA__ai_cake.json").write_text('{"ticker":"NVDA","run_id":"legacy"}')
    state = da._try_load_demo("NVDA", "ai_cake")
    assert state is not None
    assert state["run_id"] == "legacy"


def test_direct_agent_load_demo_returns_none_when_no_match(tmp_path, monkeypatch):
    """No demo for that ticker × thesis → return None so the caller
    falls back to a minimal state (and the user gets a clear soft-error)."""
    import importlib

    da = importlib.import_module("ui.pages.direct_agent")
    monkeypatch.setattr(da, "DEMO_DIR", tmp_path)
    assert da._try_load_demo("XYZ", "fake_thesis") is None


def test_direct_agent_list_other_thesis_demos_finds_alt_theses(tmp_path, monkeypatch):
    """When CEG was drilled under ai_cake but the user opens Direct Agent
    under nvda_halo, the loader must surface ai_cake as an existing
    drill-in available for CEG."""
    import importlib

    da = importlib.import_module("ui.pages.direct_agent")
    monkeypatch.setattr(da, "DEMO_DIR", tmp_path)
    (tmp_path / "CEG__ai_cake__abcdef12.json").write_text("{}")
    (tmp_path / "CEG__construction.json").write_text("{}")
    # Same ticker different thesis — both should appear.
    other = da._list_other_thesis_demos("CEG", exclude_slug="nvda_halo")
    assert "ai_cake" in other
    assert "construction" in other
    # The excluded slug must not appear even if a matching file exists.
    (tmp_path / "CEG__nvda_halo__deadbeef.json").write_text("{}")
    other2 = da._list_other_thesis_demos("CEG", exclude_slug="nvda_halo")
    assert "nvda_halo" not in other2


def test_direct_agent_list_other_thesis_demos_returns_empty_when_no_alts(
    tmp_path, monkeypatch
):
    import importlib

    da = importlib.import_module("ui.pages.direct_agent")
    monkeypatch.setattr(da, "DEMO_DIR", tmp_path)
    # No demos at all → nothing to suggest.
    assert da._list_other_thesis_demos("CEG", exclude_slug="ai_cake") == []
    # A demo for the SELECTED thesis is not surfaced as an alt.
    (tmp_path / "CEG__ai_cake__abcdef12.json").write_text("{}")
    assert da._list_other_thesis_demos("CEG", exclude_slug="ai_cake") == []


def test_direct_agent_page_renders_agent_dropdown():
    """The Agent selectbox must:
      - Exist on the page
      - Include ALL five agents (fundamentals, filings, news, risk, synthesis).
        Catches the regression where AGENT_NAMES was extended but the
        dropdown still rendered an old cached tuple — see Step 8 retro."""
    at = AppTest.from_file(
        str(PAGES_DIR / "direct_agent.py"), default_timeout=DASHBOARD_TIMEOUT_S
    )
    at.run()
    assert not at.exception
    selectboxes = at.selectbox
    agent_box = next((sb for sb in selectboxes if sb.label == "Agent"), None)
    assert agent_box is not None, (
        f"Direct Agent missing 'Agent' selectbox; saw: {[s.label for s in selectboxes]}"
    )
    options = list(agent_box.options or [])
    expected = {"fundamentals", "filings", "news", "risk", "synthesis"}
    missing = expected - set(options)
    assert not missing, (
        f"Agent dropdown missing options: {sorted(missing)}; saw: {options}"
    )


# --- New Thesis form fields -----------------------------------------------


def test_mission_control_renders_cio_telemetry_panels():
    """Step 11.19 — Mission Control must render the new CIO sections:
      - 'Recent CIO actions' (with model + cost columns)
      - 'Recent CIO cycles' (with model + total $ columns)
      - 'CIO model performance — last 30 days'
    """
    # Seed state.db with one CIO cycle + 2 actions so the panels have data.
    from data import state as state_db

    rid = state_db.start_cio_run("heartbeat")
    state_db.record_cio_action(
        ticker="NVDA", thesis="ai_cake", action="drill",
        cio_run_id=rid, model_used="openai/gpt-5.4-mini",
        tokens_in=2500, tokens_out=80, cost_usd=0.0025, latency_s=1.5,
        confidence="high",
    )
    state_db.record_cio_action(
        ticker="MSFT", thesis="ai_cake", action="dismiss",
        cio_run_id=rid, model_used="openai/gpt-5.4-mini",
        tokens_in=2200, tokens_out=70, cost_usd=0.0021, latency_s=1.2,
        confidence="high",
    )
    state_db.finish_cio_run(
        rid, "completed", model_used="openai/gpt-5.4-mini",
        total_cost_usd=0.0046, n_actions=2, n_drilled=1, n_dismissed=1,
        duration_s=2.7,
    )

    at = AppTest.from_file(
        str(PAGES_DIR / "mission_control.py"), default_timeout=DASHBOARD_TIMEOUT_S
    )
    at.run()
    assert not at.exception, f"page raised: {[e.message for e in at.exception]}"

    # Section headers present
    markdown_blob = " ".join(m.value for m in at.markdown)
    for required in (
        "Recent CIO actions",
        "Recent CIO cycles",
        "CIO model performance",
    ):
        assert required in markdown_blob, (
            f"Mission Control missing section: {required!r}"
        )


def test_theses_admin_renders_three_sections_and_per_row_buttons():
    """Theses admin page must show:
      - "Curated theses" / "Ad-hoc theses" / "Archive" subheaders
      - At least one curated thesis row + Demote button (real repo has ai_cake)
      - At least one adhoc thesis row + Promote button (real repo has adhoc_*)
    """
    at = AppTest.from_file(
        str(PAGES_DIR / "theses_admin.py"), default_timeout=DASHBOARD_TIMEOUT_S
    )
    at.run()
    assert not at.exception, f"page raised: {[e.message for e in at.exception]}"

    subheaders = [s.value for s in at.subheader]
    for expected in ("Curated theses", "Ad-hoc theses", "Archive"):
        assert expected in subheaders, (
            f"Theses admin missing '{expected}' subheader; saw: {subheaders}"
        )

    button_keys = {b.key for b in at.button if b.key}
    # At least one Demote button for a curated slug AND one Promote button
    # for an adhoc slug — otherwise the page isn't functional.
    has_demote = any(k.startswith("demote_") for k in button_keys)
    has_promote = any(k.startswith("promote_") for k in button_keys)
    assert has_demote, f"no Demote buttons rendered; keys: {button_keys}"
    assert has_promote, f"no Promote buttons rendered; keys: {button_keys}"


def test_new_thesis_page_renders_thesis_name_input():
    at = AppTest.from_file(
        str(PAGES_DIR / "new_thesis.py"), default_timeout=DASHBOARD_TIMEOUT_S
    )
    at.run()
    assert not at.exception
    text_inputs = at.text_input
    assert any(
        "Thesis name" in (ti.label or "") for ti in text_inputs
    ), f"New Thesis form missing 'Thesis name' input; saw: {[t.label for t in text_inputs]}"


# --- Render-side dollar-escape regression ----------------------------------
#
# Streamlit's markdown renderer uses KaTeX for math, which matches `$...$`
# pairs — when the synthesis report contains "the current price of $205.66 is
# below $250 target", KaTeX greedily renders the span between them as inline
# math (italic, no spaces). _md_safe escapes `$` followed by a digit so KaTeX
# leaves dollar amounts alone. This test pins the behaviour so a future edit
# can't silently regress the report text into garbage italic.


def test_md_safe_escapes_dollar_before_digit():
    from ui.app import _md_safe

    assert _md_safe(
        "current price of $205.66 is below $250 target"
    ) == "current price of \\$205.66 is below \\$250 target"


def test_md_safe_lives_in_shared_components_module():
    """Both ui/app.py AND ui/pages/direct_agent.py need the KaTeX dollar
    escape — Direct Agent's synthesis-Q&A answers also contain dollar
    amounts. The helper lives in ui/components.py as the shared home;
    ui/app.py aliases it for backwards compat."""
    from ui import components, app as app_mod

    assert components.md_safe is not None
    # ui/app.py keeps `_md_safe` as an alias to the shared function so
    # existing call sites don't break.
    assert app_mod._md_safe is components.md_safe


def test_md_safe_handles_none_input():
    """Defensive: callers might pass None when an answer string is missing."""
    from ui.components import md_safe

    assert md_safe(None) == ""
    assert md_safe("") == ""


# --- Runs tab (the former Run Inspector, Step 10c.8) -----------------------


def test_runs_tab_renders_with_empty_db(tmp_path, monkeypatch):
    """Empty state.db (no runs) → the Runs tab renders a friendly 'no runs'
    note, doesn't crash."""
    from data import state as state_db

    monkeypatch.setattr(state_db, "DB_PATH", tmp_path / "empty.db")
    state_db.init_db(db_path=tmp_path / "empty.db")

    at = AppTest.from_file(
        str(PAGES_DIR / "mission_control.py"), default_timeout=DASHBOARD_TIMEOUT_S
    )
    at.run()
    assert not at.exception
    # Should warn (not error) so the user knows what to do.
    warnings = [w.value for w in at.warning]
    assert any("No graph runs" in w for w in warnings)


def test_runs_tab_renders_with_populated_db(tmp_path, monkeypatch):
    """Populate state.db with a fake run + per-node rows; the Runs tab
    must render the run header + picker without crashing."""
    from data import state as state_db

    db = tmp_path / "test.db"
    monkeypatch.setattr(state_db, "DB_PATH", db)
    run_id = state_db.start_graph_run("NVDA", "ai_cake", db_path=db)
    state_db.record_node_run(
        run_id, "fundamentals", "2026-05-05T10:00:00+00:00", "2026-05-05T10:00:01+00:00",
        1.0, "completed", tokens_in=500, tokens_out=200, cost_usd=0.0035, n_calls=1,
        db_path=db,
    )
    state_db.record_node_run(
        run_id, "synthesis", "2026-05-05T10:00:02+00:00", "2026-05-05T10:00:15+00:00",
        13.0, "completed", tokens_in=3000, tokens_out=1500, cost_usd=0.06, n_calls=1,
        db_path=db,
    )
    state_db.finish_graph_run(run_id, status="completed", confidence="medium", db_path=db)

    at = AppTest.from_file(
        str(PAGES_DIR / "mission_control.py"), default_timeout=DASHBOARD_TIMEOUT_S
    )
    at.run()
    assert not at.exception
    # The run picker should list the run we just inserted.
    sel = next((sb for sb in at.selectbox if sb.label == "Run"), None)
    assert sel is not None
    assert any("NVDA" in opt and "ai_cake" in opt for opt in sel.options)


def test_runs_tab_langsmith_url_only_when_tracing_enabled(monkeypatch):
    """Defensive: the LangSmith deep-link button MUST NOT appear when
    LANGSMITH_TRACING is unset — there'd be no traces to link to. (Moved
    from the Run Inspector into Mission Control's Runs tab.)"""
    import importlib

    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    ri = importlib.import_module("ui.pages.mission_control")
    assert ri._langsmith_url_for_run("abc-123") is None

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_PROJECT", "finaq")
    url = ri._langsmith_url_for_run("abc-123")
    assert url is not None
    assert "finaq" in url
    assert "abc-123" in url


def test_md_safe_leaves_lone_dollar_alone():
    from ui.app import _md_safe

    # No digit after the `$` → not a dollar amount, leave it alone so genuine
    # math (rare in financial reports but possible) still renders.
    assert _md_safe("symbol $ alone") == "symbol $ alone"


def test_md_safe_idempotent_on_empty_input():
    from ui.app import _md_safe

    assert _md_safe("") == ""
    assert _md_safe("plain text no markup") == "plain text no markup"


# --- Run traceability (schema v6) ------------------------------------------


def _seed_failed_run(db, state_db) -> str:
    """One WEN run that failed on multiple fronts — the shape the runs
    table + inspector must make obvious."""
    run_id = state_db.start_graph_run("WEN", "wen", db_path=db)
    now = state_db._now_iso()
    state_db.record_node_run(
        run_id, "fundamentals", now, now, 46.0, "completed",
        tokens_in=21000, tokens_out=2400, cost_usd=0.06, n_calls=4, db_path=db,
    )
    state_db.record_node_run(
        run_id, "filings", now, now, 122.0, "failed",
        error="chroma timeout", db_path=db,
    )
    state_db.record_error("filings", "chroma timeout", run_id=run_id, db_path=db)
    state_db.record_llm_call(
        run_id=run_id, node="fundamentals", model="test/model",
        latency_s=6.1, tokens_in=5200, tokens_out=600, cost_usd=0.014,
        prompt_excerpt='[{"role": "user", "content": "kpis"}]',
        response_excerpt='{"summary": "ok"}',
        db_path=db,
    )
    state_db.finish_graph_run(run_id, "failed", duration_s=214.0, db_path=db)
    return run_id


def test_mission_control_runs_table_shows_cost_and_failed_agents(
    tmp_path, monkeypatch
):
    """The revamped runs table must surface cost + failed-agent names and
    derive the failed status label — the exact info that was missing when
    a run 'failed on multiple fronts' and looked like noise."""
    from data import state as state_db

    db = tmp_path / "mc.db"
    monkeypatch.setattr(state_db, "DB_PATH", db)
    _seed_failed_run(db, state_db)

    at = AppTest.from_file(
        str(PAGES_DIR / "mission_control.py"), default_timeout=DASHBOARD_TIMEOUT_S
    )
    at.run()
    assert not at.exception, f"page raised: {[e.message for e in at.exception]}"

    # The runs table is now bespoke HTML rows (st.markdown) with a per-row
    # Open button — not an st.dataframe. Assert on the rendered markup.
    blob = " ".join(m.value for m in at.markdown)
    assert "Spend today" in blob            # KPI card
    assert "mcrow" in blob                  # a styled run row rendered
    assert "$0.0600" in blob                # this run's cost
    assert 'spill fail' in blob             # failed-status pill
    assert 'dot fail' in blob               # filings failed → a brick dot
    # Per-row Open button opens the inspector.
    assert any((b.label or "").startswith("Open") for b in at.button), (
        f"no per-row Open button; buttons: {[b.label for b in at.button]}"
    )


def test_mission_control_degraded_status_label():
    """Completed run + failed node or soft error → 'degraded', never a
    healthy-looking 'completed'."""
    import importlib

    mc = importlib.import_module("ui.pages.mission_control")
    assert "degraded" in mc._run_status_label(
        {"status": "completed", "failed_nodes": 1, "n_errors": 0}
    )
    assert "degraded" in mc._run_status_label(
        {"status": "completed", "failed_nodes": 0, "n_errors": 2}
    )
    assert "completed" in mc._run_status_label(
        {"status": "completed", "failed_nodes": 0, "n_errors": 0}
    )
    assert "failed" in mc._run_status_label({"status": "failed"})


def test_runs_tab_renders_trace_timeline_and_banner(tmp_path, monkeypatch):
    """The Runs tab must render the failure banner, agent timeline, and
    the local LLM call trace for a failed run."""
    from data import state as state_db

    db = tmp_path / "ri.db"
    monkeypatch.setattr(state_db, "DB_PATH", db)
    _seed_failed_run(db, state_db)

    at = AppTest.from_file(
        str(PAGES_DIR / "mission_control.py"), default_timeout=DASHBOARD_TIMEOUT_S
    )
    at.run()
    assert not at.exception, f"page raised: {[e.message for e in at.exception]}"

    # Failure banner names the failed agent.
    banner = " ".join(e.value for e in at.error)
    assert "filings" in banner
    assert "1 of 2 agents failed" in banner

    # Trace + timeline sections present.
    markdown_blob = " ".join(m.value for m in at.markdown)
    assert "LLM call trace" in markdown_blob
    assert "Agent timeline" in markdown_blob
    # One expander per LLM call, labelled with the node.
    assert any("fundamentals" in (x.label or "") for x in at.expander)


def test_runs_tab_preselects_run_from_session_state(tmp_path, monkeypatch):
    """Health's row-click hands over `inspect_run_id` via session state —
    the Runs tab must open on THAT run, not the newest."""
    import time as _time

    from data import state as state_db

    db = tmp_path / "ri2.db"
    monkeypatch.setattr(state_db, "DB_PATH", db)
    older = _seed_failed_run(db, state_db)
    _time.sleep(0.02)  # distinct started_at so ordering is deterministic
    newer = state_db.start_graph_run("NU", "nu", db_path=db)
    state_db.finish_graph_run(newer, "completed", db_path=db)

    at = AppTest.from_file(
        str(PAGES_DIR / "mission_control.py"), default_timeout=DASHBOARD_TIMEOUT_S
    )
    at.session_state["inspect_run_id"] = older
    at.run()
    assert not at.exception, f"page raised: {[e.message for e in at.exception]}"
    sel = next((sb for sb in at.selectbox if sb.label == "Run"), None)
    assert sel is not None
    assert older[:8] in (sel.value or ""), (
        f"expected preselected run {older[:8]}, got {sel.value!r}"
    )


def test_runs_tab_shows_cio_trigger_line(tmp_path, monkeypatch):
    """A run drilled by the CIO must show the 'triggered by CIO' backlink
    in the header; the helper is unit-tested directly."""
    import importlib

    from data import state as state_db

    db = tmp_path / "ri3.db"
    monkeypatch.setattr(state_db, "DB_PATH", db)
    cycle = state_db.start_cio_run("heartbeat", db_path=db)
    run_id = state_db.start_graph_run("NU", "nu", db_path=db)
    state_db.record_cio_action(
        ticker="NU", thesis="nu", action="drill", cio_run_id=cycle,
        drill_run_id=run_id, confidence="high", db_path=db,
    )
    state_db.finish_graph_run(run_id, "completed", db_path=db)

    ri = importlib.import_module("ui.pages.mission_control")
    line = ri._trigger_line(run_id)
    assert "CIO heartbeat" in line
    assert "drill" in line
    # Manual runs fall back to the manual label.
    manual = state_db.start_graph_run("EME", "construction", db_path=db)
    assert "manual" in ri._trigger_line(manual)


def test_runs_tab_demo_loader_never_matches_other_runs(tmp_path, monkeypatch):
    """Regression: the payload loader's `{prefix}*.json` glob matched OTHER
    runs' suffixed files (and longer slugs sharing the prefix), silently
    rendering a different run's agent payloads. Only the exact run-id
    suffix or the exact legacy name may match."""
    import importlib

    ri = importlib.import_module("ui.pages.mission_control")
    monkeypatch.setattr(ri, "DEMO_DIR", tmp_path)

    (tmp_path / "NVDA__nu__aaaaaaaa.json").write_text('{"run_id": "other-run"}')
    (tmp_path / "NVDA__nu_extended__bbbbbbbb.json").write_text('{"run_id": "longer-slug"}')

    # Different run id, no legacy file → nothing must match.
    assert ri._load_demo_state("NVDA", "nu", "cccccccc-1234") is None

    # Exact run-id suffix matches.
    (tmp_path / "NVDA__nu__cccccccc.json").write_text('{"run_id": "mine"}')
    state = ri._load_demo_state("NVDA", "nu", "cccccccc-1234")
    assert state and state["run_id"] == "mine"

    # Legacy exact name is the only fallback.
    (tmp_path / "NVDA__nu.json").write_text('{"run_id": "legacy"}')
    state = ri._load_demo_state("NVDA", "nu", "dddddddd-0000")
    assert state and state["run_id"] == "legacy"


# --- Mission Control v2, Phase 2: Live tab, replay, redirect stub -----------


_SYSTEM_PROMPT = "You are a senior equity research analyst.\n" + ("Rule line.\n" * 400)
_USER_PROMPT = "AS OF: 2026-10-21\nTICKER: NVDA\n" + ("KPI line.\n" * 50)


def _seed_running_run(state_db) -> tuple[str, int, int]:
    """One run in flight: a finished load_thesis row, a running news node with
    an in-flight call whose full messages are stored (schema v9)."""
    run_id = state_db.start_graph_run("NVDA", "nu", source="dashboard")
    done = state_db.start_node_run(run_id, "load_thesis")
    state_db.finish_node_run(done, "completed", duration_s=0.05)
    node_id = state_db.start_node_run(run_id, "news")
    call_id = state_db.start_llm_call(
        run_id=run_id,
        node="news",
        model="test/model",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_PROMPT},
        ],
    )
    return run_id, node_id, call_id


def test_mission_control_has_live_runs_health_tabs_and_renders_the_component():
    """The page is three tabs; the Live tab mounts the component (an iframe
    element) even on an empty DB."""
    import re

    at = AppTest.from_file(
        str(PAGES_DIR / "mission_control.py"), default_timeout=DASHBOARD_TIMEOUT_S
    )
    at.run()
    assert not at.exception, f"page raised: {[e.message for e in at.exception]}"
    assert [t.label for t in at.tabs] == ["Live", "Runs", "Health"]
    assert len(at.get("iframe")) == 1, "the Live fragment must mount exactly one component"
    # Every full script run mounts the iframe anew, so it must carry a fresh
    # `now` for the component's clock offset (the memo only serves fragment
    # reruns, which do not re-mount).
    first = re.search(r'"now":"([^"]+)"', at.get("iframe")[0].proto.srcdoc).group(1)
    at.run()
    second = re.search(r'"now":"([^"]+)"', at.get("iframe")[0].proto.srcdoc).group(1)
    assert second > first


def test_live_fragment_emits_one_component_and_memoises_identical_snapshots(
    tmp_path, monkeypatch
):
    """Driven directly (run_every is inert under AppTest): one components.html
    per fragment run; an unchanged snapshot re-emits the byte-identical HTML
    string (so Streamlit keeps the iframe mounted); a changed snapshot yields
    new HTML carrying the new content."""
    import importlib

    from data import state as state_db

    mc = importlib.import_module("ui.pages.mission_control")
    monkeypatch.setattr(state_db, "DB_PATH", tmp_path / "live.db")
    mc._LIVE_CACHE.update({"hash": None, "html": None})
    emitted: list[str] = []
    monkeypatch.setattr(mc.components, "html", lambda doc, **kw: emitted.append(doc))

    run_id, node_id, call_id = _seed_running_run(state_db)

    mc.render_live_panel()
    assert len(emitted) == 1
    first = emitted[0]
    # Full prompts ride along (JSON-escaped, so match a newline-free slice).
    assert "You are a senior equity research analyst." in first and "AS OF: 2026-10-21" in first
    assert '"status":"running"' in first

    mc.render_live_panel()
    assert len(emitted) == 2
    assert emitted[1] == first, "identical snapshot must re-emit byte-identical HTML"

    state_db.finish_llm_call(
        call_id, status="completed", latency_s=0.7, tokens_in=9, tokens_out=4,
        response_text="THE ANSWER", cost_usd=0.001,
    )
    mc.render_live_panel()
    assert len(emitted) == 3
    assert emitted[2] != first
    assert "THE ANSWER" in emitted[2]


def test_live_html_substitutes_every_token_and_parses_as_html(tmp_path, monkeypatch):
    """Every {{token}} in ui/mc_live.html is known to mc_tokens() + the page,
    none survives rendering, the document parses, and a prompt containing
    `</script>` cannot break out of the JSON data block."""
    import importlib
    import re
    from html.parser import HTMLParser

    from data import state as state_db
    from ui.components import mc_tokens

    mc = importlib.import_module("ui.pages.mission_control")
    template = mc.MC_TEMPLATE.read_text(encoding="utf-8")
    placeholders = set(re.findall(r"\{\{(\w+)\}\}", template))
    known = set(mc_tokens()) | {"snapshot_json", "mode"}
    assert placeholders <= known, f"unknown placeholders: {placeholders - known}"
    for required in ("bg", "accent", "ink", "night_bg", "theme", "mode", "snapshot_json"):
        assert required in placeholders

    monkeypatch.setattr(state_db, "DB_PATH", tmp_path / "t.db")
    run_id = state_db.start_graph_run("NVDA", "nu", source="cli")
    state_db.start_node_run(run_id, "risk")
    state_db.start_llm_call(
        run_id=run_id, node="risk", model="m",
        messages=[
            {"role": "system", "content": "S uses {{mode}} and {{theme}} literally"},
            {"role": "user", "content": "</script><b>x"},
        ],
    )
    for mode in ("live", "replay"):
        snap = state_db.live_snapshot() if mode == "live" else mc.replay_snapshot(run_id)
        doc = mc.live_html(snap, mode=mode)
        # Stored text keeps its own braces verbatim: the data block is substituted last.
        assert "S uses {{mode}} and {{theme}} literally" in doc
        assert not re.search(r"\{\{\w+\}\}", doc.replace("{{mode}} and {{theme}}", "")), (
            "unsubstituted placeholder left in the HTML"
        )
        assert "</script><b>x" not in doc and "\\u003c/script>" in doc

        class _Parser(HTMLParser):
            tags: list[str] = []

            def handle_starttag(self, tag, attrs):
                self.tags.append(tag)

        parser = _Parser()
        parser.tags = []
        parser.feed(doc)
        assert {"html", "head", "style", "body", "script"} <= set(parser.tags)
        assert f'data-mode="{mode}"' in doc


def test_mc_tokens_day_is_the_brand_palette_and_night_is_off_by_default():
    from ui.components import INK, PARCHMENT, SAGE, mc_tokens

    day = mc_tokens()
    assert (day["bg"], day["accent"], day["ink"], day["theme"]) == (PARCHMENT, SAGE, INK, "day")
    assert day["night_bg"] != day["bg"] and day["night_accent"] != day["accent"]
    assert mc_tokens("night")["theme"] == "night"
    assert mc_tokens("anything-else")["theme"] == "day"


def test_runs_tab_shows_full_system_and_user_prompt_for_a_call(tmp_path, monkeypatch):
    """The LLM call trace renders the FULL stored messages (not the 2,000-char
    excerpt): a system prompt longer than the excerpt cap must appear whole in
    an st.code block, alongside the user prompt and the response."""
    from data import state as state_db

    monkeypatch.setattr(state_db, "DB_PATH", tmp_path / "full.db")
    run_id, node_id, call_id = _seed_running_run(state_db)
    state_db.finish_llm_call(
        call_id, status="completed", latency_s=1.2, tokens_in=10, tokens_out=5,
        response_text="FULL RESPONSE TEXT",
    )
    state_db.finish_node_run(node_id, "completed", duration_s=1.3, n_calls=1)
    state_db.finish_graph_run(run_id, "completed", duration_s=2.0)
    assert len(_SYSTEM_PROMPT) > state_db.LLM_EXCERPT_MAX_CHARS

    at = AppTest.from_file(
        str(PAGES_DIR / "mission_control.py"), default_timeout=DASHBOARD_TIMEOUT_S
    )
    at.run()
    assert not at.exception, f"page raised: {[e.message for e in at.exception]}"
    codes = [c.value for c in at.code]  # st.code dedents + strips its body
    assert _SYSTEM_PROMPT.strip() in codes, "full system prompt must be rendered verbatim"
    assert _USER_PROMPT.strip() in codes, "full user prompt must be rendered verbatim"
    assert "FULL RESPONSE TEXT" in codes
    assert any("call 1 — news" in (x.label or "") for x in at.expander)


def test_run_id_query_param_preselects_run_on_mission_control(tmp_path, monkeypatch):
    """`?run_id=` (Telegram deep link) selects that run in the Runs tab even
    when it is not the newest."""
    import time as _time

    from data import state as state_db

    db = tmp_path / "deep.db"
    monkeypatch.setattr(state_db, "DB_PATH", db)
    older = _seed_failed_run(db, state_db)
    _time.sleep(0.02)
    newer = state_db.start_graph_run("NU", "nu", db_path=db)
    state_db.finish_graph_run(newer, "completed", db_path=db)

    at = AppTest.from_file(
        str(PAGES_DIR / "mission_control.py"), default_timeout=DASHBOARD_TIMEOUT_S
    )
    at.query_params["run_id"] = older
    at.run()
    assert not at.exception, f"page raised: {[e.message for e in at.exception]}"
    sel = next((sb for sb in at.selectbox if sb.label == "Run"), None)
    assert sel is not None and older[:8] in (sel.value or "")
    assert at.session_state["mc_tabs"] == "Runs"  # the deep link opens the Runs tab


def test_run_inspector_stub_redirects_keeping_run_id(monkeypatch):
    """The old page is a redirect: it carries `run_id` over as the query
    param of the switch AND as the `inspect_run_id` session key."""
    import streamlit as st

    calls: list[tuple] = []

    def _fake_switch(page, *, query_params=None):
        calls.append((page, query_params))

    monkeypatch.setattr(st, "switch_page", _fake_switch)
    at = AppTest.from_file(
        str(PAGES_DIR / "run_inspector.py"), default_timeout=DASHBOARD_TIMEOUT_S
    )
    at.query_params["run_id"] = "abc-123"
    at.run()
    assert not at.exception, f"stub raised: {[e.message for e in at.exception]}"
    assert calls == [("pages/mission_control.py", {"run_id": "abc-123"})]
    assert at.session_state["inspect_run_id"] == "abc-123"
    assert at.session_state["mc_jump_tab"] == "Runs"


def test_replay_snapshot_wraps_a_run_with_its_cio_cycle(tmp_path, monkeypatch):
    """Replay feeds the component the live snapshot's shape: the run with full
    messages under `runs`, the cycle behind it (spans + decisions) under
    `cio`, `now` at the run's end."""
    import importlib

    from data import state as state_db

    mc = importlib.import_module("ui.pages.mission_control")
    monkeypatch.setattr(state_db, "DB_PATH", tmp_path / "replay.db")
    cycle = state_db.start_cio_run("on_demand")
    with state_db.span("cio_run", cycle, "decide", label="NVDA"):
        pass
    run_id, node_id, call_id = _seed_running_run(state_db)
    state_db.finish_llm_call(call_id, status="completed", latency_s=1.0, response_text="R")
    state_db.finish_node_run(node_id, "completed", duration_s=1.0, n_calls=1)
    state_db.finish_graph_run(run_id, "completed", duration_s=2.0)
    state_db.record_cio_action(
        ticker="NVDA", thesis="nu", action="drill", cio_run_id=cycle, drill_run_id=run_id,
        trigger="on_demand", confidence="high",
    )
    state_db.finish_cio_run(cycle, "completed", n_actions=1, n_drilled=1)

    snap = mc.replay_snapshot(run_id)
    assert snap is not None and [r["run_id"] for r in snap["runs"]] == [run_id]
    [cio] = snap["cio"]
    assert cio["run_id"] == cycle and [s["name"] for s in cio["steps"]] == ["decide"]
    assert cio["actions"][0]["drill_run_id"] == run_id
    assert snap["now"] == snap["runs"][0]["ended_at"]
    [node] = [n for n in snap["runs"][0]["nodes"] if n["node"] == "news"]
    assert node["calls"][0]["messages"][0]["text"] == _SYSTEM_PROMPT
    assert mc.replay_snapshot("missing") is None


def test_health_open_button_jumps_to_runs_tab_with_that_run(tmp_path, monkeypatch):
    """Health's per-row "Open →" stages a jump: the next run opens the Runs tab
    with THAT run selected (not the newest) and the URL carries its id; a later
    manual pick survives a plain rerun."""
    import time as _time

    from data import state as state_db

    db = tmp_path / "jump.db"
    monkeypatch.setattr(state_db, "DB_PATH", db)
    older = state_db.start_graph_run("AAA", "t1", db_path=db)
    state_db.finish_graph_run(older, "completed", db_path=db)
    _time.sleep(0.02)
    newer = state_db.start_graph_run("BBB", "t2", db_path=db)
    state_db.finish_graph_run(newer, "completed", db_path=db)

    at = AppTest.from_file(
        str(PAGES_DIR / "mission_control.py"), default_timeout=DASHBOARD_TIMEOUT_S
    )
    at.session_state["mc_tabs"] = "Health"
    at.run()
    assert not at.exception, f"page raised: {[e.message for e in at.exception]}"
    open_btn = next(b for b in at.button if b.key == f"open_{older}")
    open_btn.click().run()
    assert not at.exception, f"page raised: {[e.message for e in at.exception]}"
    assert at.session_state["mc_tabs"] == "Runs"
    sel = next(sb for sb in at.selectbox if sb.label == "Run")
    assert older[:8] in (sel.value or "")
    assert at.query_params.get("run_id") == [older]  # AppTest stores parse_qs lists

    sel.select(next(o for o in sel.options if newer[:8] in o)).run()
    at.run()  # a plain rerun (e.g. a tab click) keeps the manual pick
    sel = next(sb for sb in at.selectbox if sb.label == "Run")
    assert newer[:8] in (sel.value or "")

