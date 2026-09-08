"""Freshness gate — is the local ChromaDB ingest up-to-date for a ticker?

Composes `data.edgar.latest_filing_dates` (SEC's authoritative recent-filings
index) with `data.chroma.last_filings_by_type` (what's actually in the local
ChromaDB) and returns a structured diff. Consumed by:

  - the Streamlit `Run drill-in` button — opens a confirmation dialog when
    stale (Ingest + drill / Cancel)
  - the Telegram `/drill` handler — replies with an inline-keyboard
    confirmation
  - the CIO heartbeat — probes every candidate ticker (the report feeds
    the planner as `edgar_freshness`); auto-ingests only anchor tickers,
    on-demand `/cio` tickers, and tickers the planner picks for a drill

Soft-fail by design: any EDGAR error returns a report with `is_stale=False`
and `edgar_error` populated, so callers drill on whatever's in the local
corpus rather than blocking on a flaky network. ChromaDB read failures
yield empty per-form data but never raise.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date

from data.chroma import last_filings_by_type
from data.edgar import latest_filing_dates
from utils import logger

# Forms we care about. Matches DEFAULT_LIMITS in data/edgar.py so the
# freshness check and the ingest pipeline agree on what's "tracked".
TRACKED_FORMS: tuple[str, ...] = ("10-K", "10-Q", "20-F", "6-K")


@dataclass(frozen=True)
class FormDiff:
    """Per-form freshness — what EDGAR has vs what's ingested in ChromaDB."""

    form: str
    edgar_date: str | None  # ISO YYYY-MM-DD; None when EDGAR has no filings of this form
    chroma_date: str | None  # ISO YYYY-MM-DD; None when ChromaDB has none
    behind_days: int  # 0 when fresh or unknown; positive when ChromaDB trails EDGAR

    @property
    def is_missing(self) -> bool:
        """EDGAR has filings of this form but ChromaDB has none ingested."""
        return self.edgar_date is not None and self.chroma_date is None

    @property
    def is_behind(self) -> bool:
        """ChromaDB has filings of this form but EDGAR has a newer one."""
        return (
            self.edgar_date is not None
            and self.chroma_date is not None
            and self.edgar_date > self.chroma_date
        )


@dataclass(frozen=True)
class FreshnessReport:
    """Whole-ticker freshness summary used to drive the drill-time gate."""

    ticker: str
    is_stale: bool  # True when any tracked form is missing or behind at EDGAR
    per_form: list[FormDiff]  # one entry per requested form, preserving input order
    edgar_error: str | None = None  # populated when EDGAR lookup failed

    def stale_forms(self) -> list[FormDiff]:
        """Subset of `per_form` that drove `is_stale=True`. Used by the UI
        to render only the rows the user needs to act on."""
        return [d for d in self.per_form if d.is_missing or d.is_behind]


def _days_between(later: str, earlier: str) -> int:
    """Calendar days between two ISO dates (later - earlier). Returns 0 when
    either input is missing or unparseable — callers should not interpret 0
    as "same date" without also checking the input strings."""
    if not later or not earlier:
        return 0
    try:
        d_later = date.fromisoformat(later)
        d_earlier = date.fromisoformat(earlier)
    except ValueError:
        return 0
    return (d_later - d_earlier).days


def check_ingest_freshness(
    ticker: str,
    forms: tuple[str, ...] = TRACKED_FORMS,
) -> FreshnessReport:
    """Compare EDGAR's recent index to local ChromaDB ingest for `ticker`.

    A ticker is `stale` when EDGAR reports a `filingDate` strictly greater
    than the local ChromaDB `filed_date` for any form in `forms` — OR when
    EDGAR has any of those forms and ChromaDB has none. Forms with no EDGAR
    presence (e.g., 20-F for a domestic filer) never drive staleness.

    `edgar_error` populated means freshness is unknown; `is_stale` is False
    in that case so a transient SEC outage doesn't block drills.

    FINAQ_SKIP_FRESHNESS_PROBES short-circuits the whole check to the same
    "unknown, not stale" report — without touching EDGAR or the ChromaDB
    Rust client (see the segfault note in data/chroma.py + POSTPONED §2).
    Gating here, not just in the chroma probes, matters: an empty chroma
    result with a live EDGAR date would otherwise read as "stale" and
    funnel every drill into the ingest path the switch exists to avoid.
    """
    if os.getenv("FINAQ_SKIP_FRESHNESS_PROBES"):
        return FreshnessReport(
            ticker=ticker.upper(),
            is_stale=False,
            per_form=[],
            edgar_error="freshness probes disabled (FINAQ_SKIP_FRESHNESS_PROBES)",
        )

    edgar = latest_filing_dates(ticker, forms=forms)
    chroma = last_filings_by_type(ticker)

    edgar_error: str | None = None
    if not edgar:
        edgar_error = "EDGAR submissions lookup returned no data"

    diffs: list[FormDiff] = []
    is_stale = False
    for form in forms:
        e_date = edgar.get(form) or None
        c_date = chroma.get(form) or None
        behind = _days_between(e_date or "", c_date or "") if (e_date and c_date) else 0
        if e_date and (c_date is None or e_date > c_date):
            is_stale = True
        diffs.append(
            FormDiff(form=form, edgar_date=e_date, chroma_date=c_date, behind_days=behind)
        )

    report = FreshnessReport(
        ticker=ticker.upper(),
        is_stale=is_stale and edgar_error is None,
        per_form=diffs,
        edgar_error=edgar_error,
    )
    if report.is_stale:
        logger.info(
            f"[freshness] {ticker}: stale — "
            + ", ".join(
                f"{d.form}({d.chroma_date or 'missing'}→{d.edgar_date})"
                for d in report.stale_forms()
            )
        )
    return report
