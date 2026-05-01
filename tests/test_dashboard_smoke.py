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
    # Should now show a metric for total runs (>=1)
    metrics = at.metric
    total_runs = next(
        (m for m in metrics if "Total" in (m.label or "")),
        None,
    )
    assert total_runs is not None
    # value is a string formatted by st.metric
    assert int(total_runs.value) >= 1


# --- Direct Agent page selectors ------------------------------------------


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


def test_md_safe_leaves_lone_dollar_alone():
    from ui.app import _md_safe

    # No digit after the `$` → not a dollar amount, leave it alone so genuine
    # math (rare in financial reports but possible) still renders.
    assert _md_safe("symbol $ alone") == "symbol $ alone"


def test_md_safe_idempotent_on_empty_input():
    from ui.app import _md_safe

    assert _md_safe("") == ""
    assert _md_safe("plain text no markup") == "plain text no markup"
