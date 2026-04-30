"""Shared logging configuration and the tenacity retry decorator used by every external-call site."""

from __future__ import annotations

import logging
import os

import httpx
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s | %(message)s"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("finaq")

RETRYABLE_EXCEPTIONS = (
    httpx.HTTPError,
    httpx.TimeoutException,
    ConnectionError,
    OSError,
    TimeoutError,
)

tenacity_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    reraise=True,
)


# --- Number formatting -----------------------------------------------------


def humanize_amount(n: float, *, prefix: str = "$", precision: int = 2) -> str:
    """Format a large number with B/M/K suffix for readable display.

    Examples:
        humanize_amount(215_938_000_000) → "$215.94B"
        humanize_amount(30_000_000_000) → "$30.00B"
        humanize_amount(24_300_000_000, prefix="") → "24.30B"
        humanize_amount(1_500_000) → "$1.50M"
        humanize_amount(213.17) → "$213.17"
        humanize_amount(None) → "—"

    Used by the dashboard's KPI grid + the PDF cover-page table so the same
    formatting convention is applied everywhere.
    """
    if n is None:
        return "—"
    try:
        value = float(n)
    except (TypeError, ValueError):
        return str(n)
    if value != value:  # NaN
        return "—"
    abs_v = abs(value)
    if abs_v >= 1e12:
        return f"{prefix}{value / 1e12:,.{precision}f}T"
    if abs_v >= 1e9:
        return f"{prefix}{value / 1e9:,.{precision}f}B"
    if abs_v >= 1e6:
        return f"{prefix}{value / 1e6:,.{precision}f}M"
    if abs_v >= 1e3 and prefix == "":
        # Only abbreviate to K when no currency prefix (typical for "shares").
        # Currency in the thousands is more readable as $1,234 than $1.23K.
        return f"{value / 1e3:,.{precision}f}K"
    return f"{prefix}{value:,.{precision}f}"
