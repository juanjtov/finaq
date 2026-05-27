"""SEC EDGAR filing fetcher.

Wraps `sec-edgar-downloader` with an idempotent `download_filings` that always
fetches the 2 most recent annual reports and 4 most recent interim reports per
ticker by default — covering both domestic filers (10-K + 10-Q) and foreign
private issuers (20-F + 6-K). Also exposes `parse_filed_date` which extracts
the SEC-reported filing date from a submission's SGML header — used by
ChromaDB metadata for freshness.

For tickers that file only one of the two (e.g. NVDA has no 20-Fs, NU has
no 10-Ks), `dl.get(kind, ticker)` is a no-op when no filings exist of that
kind. Cost is one wasted API call per missing kind, which EDGAR handles fine.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path

import httpx
from sec_edgar_downloader import Downloader

from utils import logger, tenacity_retry

EDGAR_DIR = Path("data_cache/edgar")
# Domestic filers (10-K + 10-Q) and foreign private issuers (20-F + 6-K) are
# both covered. 6-K filings are press releases / interim disclosures attached
# as exhibits — they don't follow Item-X structure so each lands as one
# "misc" chunk pile via the fallback in data/chroma.py:_split_into_items.
DEFAULT_LIMITS: dict[str, int] = {"10-K": 2, "10-Q": 4, "20-F": 2, "6-K": 4}

# SEC SGML headers contain a "FILED AS OF DATE:	YYYYMMDD" line within the first
# ~50 lines of full-submission.txt.
_FILED_DATE_RE = re.compile(r"FILED AS OF DATE:\s*(\d{8})")
_HEADER_SCAN_LINES = 60


def parse_filed_date(filing_path: Path) -> str | None:
    """Extract the SEC-reported filing date from an SGML submission.

    Returns an ISO date string (YYYY-MM-DD) or None if not parseable.
    """
    try:
        with open(filing_path, encoding="utf-8", errors="ignore") as f:
            head = "".join(line for _, line in zip(range(_HEADER_SCAN_LINES), f, strict=False))
    except OSError:
        return None
    m = _FILED_DATE_RE.search(head)
    if not m:
        return None
    raw = m.group(1)  # e.g. "20240221"
    if len(raw) != 8:
        return None
    return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"


def _parse_user_agent() -> tuple[str, str]:
    ua = os.environ.get("SEC_EDGAR_USER_AGENT", "").strip()
    if not ua:
        raise RuntimeError("SEC_EDGAR_USER_AGENT is not set. See .env.example.")
    parts = ua.split(maxsplit=1)
    company = parts[0]
    email = parts[1] if len(parts) > 1 else "user@example.com"
    return company, email


def _filings_dir(ticker: str, kind: str) -> Path:
    # sec-edgar-downloader saves to {root}/sec-edgar-filings/{ticker}/{kind}/{accession}/
    return EDGAR_DIR / "sec-edgar-filings" / ticker / kind


def has_filings_in_unsupported_kinds(ticker: str) -> list[str]:
    """Return any filing-kind directories on disk for this ticker that AREN'T
    in our supported set.

    Used by the Filings agent / dashboard ingest banner to detect cases where
    EDGAR returned only filing kinds we don't ingest — historically that meant
    foreign-issuer 20-F/6-K, but those are now first-class. Today the function
    is mostly a future hook for kinds we may still skip (8-K, S-1, etc).

    Returns the unsupported kinds present, or `[]` if the ticker either has
    no EDGAR cache OR has only-supported kinds.
    """
    ticker_root = EDGAR_DIR / "sec-edgar-filings" / ticker.upper()
    if not ticker_root.exists():
        return []
    supported = set(DEFAULT_LIMITS.keys())
    return sorted(
        p.name
        for p in ticker_root.iterdir()
        if p.is_dir() and p.name not in supported and any(p.iterdir())
    )


def _existing_filings(ticker: str, kind: str, *, as_of: str | None = None) -> list[Path]:
    """Return on-disk filing paths for (ticker, kind), most-recent-first.

    Backtest mode: when `as_of="YYYY-MM-DD"` is set, drop any filing whose
    SGML-header `FILED AS OF DATE` is AFTER `as_of`. A filing the SEC accepted
    on 2025-11-12 wasn't visible to investors on 2025-09-05 — including it
    would leak future information into the backtest.

    Filings without parseable filed_dates are kept in production mode but
    DROPPED in backtest mode (better to err on the side of completeness for
    live runs and on the side of safety for backtest).
    """
    folder = _filings_dir(ticker, kind)
    if not folder.exists():
        return []
    paths = sorted(folder.glob("*/full-submission.txt"))
    if as_of is None:
        return paths

    cutoff = as_of  # ISO YYYY-MM-DD; lexicographic compare matches calendar order
    kept: list[Path] = []
    for p in paths:
        filed = parse_filed_date(p)
        if filed is None:
            # Conservative for backtest: skip filings we can't date-stamp.
            logger.debug(f"[edgar] skipping {p} in backtest mode — no parseable filed_date")
            continue
        if filed <= cutoff:
            kept.append(p)
        else:
            logger.debug(f"[edgar] excluding {p} (filed {filed} > as_of {cutoff})")
    return kept


@tenacity_retry
def _fetch_kind(ticker: str, kind: str, limit: int) -> None:
    company, email = _parse_user_agent()
    EDGAR_DIR.mkdir(parents=True, exist_ok=True)
    dl = Downloader(company, email, str(EDGAR_DIR))
    dl.get(kind, ticker, limit=limit)


def _download_sync(
    ticker: str,
    limits: dict[str, int],
    *,
    as_of: str | None = None,
    force_refresh: bool = False,
) -> list[Path]:
    paths: list[Path] = []
    for kind, limit in limits.items():
        # Production: count any on-disk filings toward `limit`. Backtest:
        # only count filings dated ≤ as_of toward `limit` so we don't
        # short-circuit out early when post-as_of filings exist on disk.
        existing = _existing_filings(ticker, kind, as_of=as_of)
        if len(existing) >= limit and not force_refresh:
            # `force_refresh=True` skips this short-circuit so a stale ticker
            # (on-disk count already at the limit, but EDGAR has a newer
            # accession) can pick up the new filing. The sec-edgar-downloader
            # library is itself idempotent on accessions — re-running it
            # won't redownload existing ones — so force_refresh's only cost
            # is one extra metadata round-trip when there's nothing new.
            mode = f" (as_of={as_of})" if as_of else ""
            logger.info(
                f"{ticker} {kind}: {len(existing)} on disk{mode} (>= {limit}), "
                f"skipping fetch"
            )
            paths.extend(existing[:limit])
            continue
        if as_of is None:
            try:
                logger.info(f"{ticker} {kind}: fetching {limit} from EDGAR")
                _fetch_kind(ticker, kind, limit)
            except Exception as e:
                logger.error(f"{ticker} {kind}: fetch failed after retries: {e}")
            # Always re-scan disk: even on partial failure we want what landed.
            paths.extend(_existing_filings(ticker, kind)[:limit])
        else:
            # Backtest mode: never fetch fresh from EDGAR. Doing so would
            # download filings that exist TODAY which by definition include
            # post-as_of content. We use whatever pre-as_of filings already
            # landed on disk via earlier production runs — and warn loudly
            # if there aren't enough.
            if len(existing) < limit:
                logger.warning(
                    f"{ticker} {kind}: only {len(existing)} pre-as_of {as_of} filings "
                    f"on disk (wanted {limit}). Backtest will run with reduced "
                    f"corpus. Re-run a production drill BEFORE the as_of date "
                    f"if you need more historical coverage."
                )
            paths.extend(existing[:limit])
    return paths


async def download_filings(
    ticker: str,
    limits: dict[str, int] | None = None,
    *,
    as_of: str | None = None,
    force_refresh: bool = False,
) -> list[Path]:
    """Download recent SEC filings. Idempotent: skips if already on disk.

    Returns a list of paths to `full-submission.txt` files actually present on disk.
    Errors are logged, not raised — caller gets whatever was successfully fetched.

    Backtest mode (`as_of="YYYY-MM-DD"`): never fetches fresh from EDGAR (any
    fetch today returns post-as_of filings). Returns only filings whose
    SGML-header `FILED AS OF DATE` ≤ as_of. Logs a warning when corpus is
    thinner than `limits` requests.

    `force_refresh=True`: bypass the "skip if on-disk count >= limit" guard
    so a stale ticker (corpus at the limit but EDGAR has a newer accession)
    actually fetches the new filing. Used by the drill-time freshness gate.
    Ignored in backtest mode — re-fetching today would leak post-as_of data.
    """
    return await asyncio.to_thread(
        _download_sync,
        ticker,
        limits or DEFAULT_LIMITS,
        as_of=as_of,
        force_refresh=force_refresh,
    )


# --- Submissions index / freshness probe -----------------------------------
# SEC's submissions endpoint returns the full recent-filings index for a
# company keyed by CIK. One ~50KB JSON round-trip tells us every recent
# filing's `form` + `filingDate` + `accessionNumber`; no filing download
# needed. The freshness gate uses this to decide whether ChromaDB ingest is
# current before a drill-in fires.

_CIK_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_CIK_MAP_CACHE_PATH = EDGAR_DIR / "cik_map.json"
_CIK_MAP_TTL_S = 7 * 86400  # weekly refresh — companies + tickers change slowly
# SEC's documented rate ceiling is 10 req/s. A 10s timeout is generous for
# both the ~1MB tickers map and the ~50KB per-company submissions JSON.
_SEC_HTTP_TIMEOUT = 10.0


def _sec_headers() -> dict[str, str]:
    """SEC requires a descriptive User-Agent identifying the operator.
    Re-uses SEC_EDGAR_USER_AGENT so the submissions API and the downloader
    agree on identity.
    """
    company, email = _parse_user_agent()
    return {"User-Agent": f"{company} {email}", "Accept-Encoding": "gzip, deflate"}


@tenacity_retry
def _fetch_cik_map() -> dict[str, str]:
    """Fetch SEC's ticker→CIK lookup table and cache it on disk.

    `company_tickers.json` is keyed by integer rank (`"0"`, `"1"`, ...) with
    `{cik_str, ticker, title}` per entry. We invert to `{TICKER: '0001234567'}`
    so callers can do a single dict lookup. Tenacity retries cover transient
    5xx / connection errors.
    """
    EDGAR_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=_SEC_HTTP_TIMEOUT, headers=_sec_headers()) as client:
        r = client.get(_CIK_MAP_URL)
        r.raise_for_status()
        payload = r.json()
    out: dict[str, str] = {}
    for entry in payload.values():
        ticker = str(entry.get("ticker", "")).upper().strip()
        cik_int = entry.get("cik_str")
        if ticker and cik_int is not None:
            out[ticker] = f"{int(cik_int):010d}"
    _CIK_MAP_CACHE_PATH.write_text(json.dumps(out))
    logger.info(f"[edgar] cached SEC ticker→CIK map ({len(out)} entries)")
    return out


def _load_cik_map() -> dict[str, str]:
    """Return the ticker→CIK map, refreshing from SEC if local copy is stale.

    Falls back to last-cached map if the SEC request fails — the map changes
    slowly so a stale map is still useful for unblocking drills during
    transient outages.
    """
    if _CIK_MAP_CACHE_PATH.exists():
        age_s = time.time() - _CIK_MAP_CACHE_PATH.stat().st_mtime
        if age_s < _CIK_MAP_TTL_S:
            try:
                return json.loads(_CIK_MAP_CACHE_PATH.read_text())
            except (json.JSONDecodeError, OSError):
                pass  # fall through to refresh
    try:
        return _fetch_cik_map()
    except Exception as e:
        logger.warning(f"[edgar] CIK map refresh failed: {e}")
        if _CIK_MAP_CACHE_PATH.exists():
            try:
                return json.loads(_CIK_MAP_CACHE_PATH.read_text())
            except Exception:
                pass
        return {}


def ticker_to_cik(ticker: str) -> str | None:
    """Return the 10-digit zero-padded CIK for `ticker`, or None if unknown.

    Required bridge because SEC's submissions endpoint is CIK-keyed, not
    ticker-keyed. Case-insensitive lookup.
    """
    if not ticker:
        return None
    return _load_cik_map().get(ticker.upper())


@tenacity_retry
def _fetch_submissions(cik: str) -> dict:
    """Fetch SEC's per-company submissions JSON. `cik` is zero-padded 10-digit."""
    url = _SUBMISSIONS_URL_TEMPLATE.format(cik=int(cik))
    with httpx.Client(timeout=_SEC_HTTP_TIMEOUT, headers=_sec_headers()) as client:
        r = client.get(url)
        r.raise_for_status()
        return r.json()


def latest_filing_dates(
    ticker: str,
    forms: tuple[str, ...] = ("10-K", "10-Q", "20-F", "6-K"),
) -> dict[str, str]:
    """Return {form: most-recent-`filingDate`} from SEC's submissions index.

    Only forms in `forms` are kept; everything else is ignored. Returns an
    empty dict on any error (unknown ticker, network failure, malformed
    response) — callers should treat empty as "freshness unknown" and not
    block on it.

    Implementation: SEC's submissions JSON exposes `filings.recent` as
    parallel arrays — `form[i]`, `filingDate[i]`, `accessionNumber[i]`. We
    zip + filter + keep the max `filingDate` (ISO YYYY-MM-DD, lexicographic-
    sortable) per form within the whitelist.
    """
    cik = ticker_to_cik(ticker)
    if cik is None:
        logger.warning(f"[edgar.latest_filing_dates] no CIK for ticker {ticker!r}")
        return {}
    try:
        payload = _fetch_submissions(cik)
    except Exception as e:
        logger.warning(
            f"[edgar.latest_filing_dates] submissions fetch failed for {ticker}: {e}"
        )
        return {}
    recent = (payload.get("filings") or {}).get("recent") or {}
    form_list = recent.get("form") or []
    date_list = recent.get("filingDate") or []
    if not form_list or len(form_list) != len(date_list):
        logger.warning(
            f"[edgar.latest_filing_dates] {ticker}: malformed submissions payload"
        )
        return {}
    keep = set(forms)
    latest: dict[str, str] = {}
    for form, fdate in zip(form_list, date_list, strict=False):
        if form not in keep or not fdate:
            continue
        if fdate > latest.get(form, ""):
            latest[form] = fdate
    return latest
