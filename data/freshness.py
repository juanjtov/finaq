"""Freshness gate — is the local filings-index ingest up-to-date for a ticker?

Composes `data.edgar.latest_filing_dates` (SEC's authoritative recent-filings
index) with `data.vectors.last_filings_by_type` (what the ingest manifest in
state.db says is indexed) and returns a structured diff. Consumed by:

  - the Streamlit `Run drill-in` button — opens a confirmation dialog when
    stale (Ingest + drill / Cancel)
  - the Telegram `/drill` handler — replies with an inline-keyboard
    confirmation
  - the CIO heartbeat — probes every candidate ticker (the report feeds
    the planner as `edgar_freshness`); auto-ingests only anchor tickers,
    on-demand `/cio` tickers, and tickers the planner picks for a drill

Soft-fail by design: any EDGAR error returns a report with `is_stale=False`
and `edgar_error` populated, so callers drill on whatever's in the local
corpus rather than blocking on a flaky network. Manifest read failures
yield empty per-form data but never raise.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date

from data.edgar import latest_filing_dates
from data.vectors import last_filings_by_type
from utils import logger

# Forms we care about. Matches DEFAULT_LIMITS in data/edgar.py so the
# freshness check and the ingest pipeline agree on what's "tracked".
TRACKED_FORMS: tuple[str, ...] = ("10-K", "10-Q", "20-F", "6-K")


@dataclass(frozen=True)
class FormDiff:
    """Per-form freshness — what EDGAR has vs what's ingested."""

    form: str
    edgar_date: str | None  # ISO YYYY-MM-DD; None when EDGAR has no filings of this form
    ingested_date: str | None  # ISO YYYY-MM-DD; None when nothing of this form is ingested
    behind_days: int  # 0 when fresh or unknown; positive when the ingest trails EDGAR

    @property
    def is_missing(self) -> bool:
        """EDGAR has filings of this form but none is ingested."""
        return self.edgar_date is not None and self.ingested_date is None

    @property
    def is_behind(self) -> bool:
        """This form is ingested but EDGAR has a newer one."""
        return (
            self.edgar_date is not None
            and self.ingested_date is not None
            and self.edgar_date > self.ingested_date
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
    """Compare EDGAR's recent index to the ingest manifest for `ticker`.

    A ticker is `stale` when EDGAR reports a `filingDate` strictly greater
    than the ingested `filed_date` for any form in `forms` — OR when EDGAR
    has any of those forms and none is ingested. Forms with no EDGAR
    presence (e.g., 20-F for a domestic filer) never drive staleness.

    `edgar_error` populated means freshness is unknown; `is_stale` is False
    in that case so a transient SEC outage doesn't block drills.

    FINAQ_SKIP_FRESHNESS_PROBES short-circuits the whole check to the same
    "unknown, not stale" report without calling EDGAR — an operator
    kill-switch for the drill-time gate. Gating here, not just in the
    manifest probe, matters: an empty manifest with a live EDGAR date would
    otherwise read as "stale" and funnel every drill into the ingest path.
    """
    if os.getenv("FINAQ_SKIP_FRESHNESS_PROBES"):
        return FreshnessReport(
            ticker=ticker.upper(),
            is_stale=False,
            per_form=[],
            edgar_error="freshness probes disabled (FINAQ_SKIP_FRESHNESS_PROBES)",
        )

    edgar = latest_filing_dates(ticker, forms=forms)
    ingested = last_filings_by_type(ticker)

    edgar_error: str | None = None
    if not edgar:
        edgar_error = "EDGAR submissions lookup returned no data"

    diffs: list[FormDiff] = []
    is_stale = False
    for form in forms:
        e_date = edgar.get(form) or None
        c_date = ingested.get(form) or None
        behind = _days_between(e_date or "", c_date or "") if (e_date and c_date) else 0
        if e_date and (c_date is None or e_date > c_date):
            is_stale = True
        diffs.append(
            FormDiff(form=form, edgar_date=e_date, ingested_date=c_date, behind_days=behind)
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
                f"{d.form}({d.ingested_date or 'missing'}→{d.edgar_date})"
                for d in report.stale_forms()
            )
        )
    return report
