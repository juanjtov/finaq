"""Step 2 unit tests — pure-logic checks, no network."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# --- data/edgar.py -----------------------------------------------------------


def test_edgar_parse_user_agent_company_and_email(monkeypatch):
    from data.edgar import _parse_user_agent

    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "FINAQ/0.1 dev@example.com")
    company, email = _parse_user_agent()
    assert company == "FINAQ/0.1"
    assert email == "dev@example.com"


def test_edgar_parse_user_agent_raises_when_missing(monkeypatch):
    from data.edgar import _parse_user_agent

    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)
    with pytest.raises(RuntimeError, match="SEC_EDGAR_USER_AGENT"):
        _parse_user_agent()


def test_edgar_existing_filings_lists_full_submissions(tmp_path, monkeypatch):
    from data import edgar

    monkeypatch.setattr(edgar, "EDGAR_DIR", tmp_path)
    base = tmp_path / "sec-edgar-filings" / "NVDA" / "10-K"
    for accession in ("0001-23-000001", "0001-23-000002"):
        (base / accession).mkdir(parents=True)
        (base / accession / "full-submission.txt").write_text("stub")
    paths = edgar._existing_filings("NVDA", "10-K")
    assert len(paths) == 2
    assert all(p.name == "full-submission.txt" for p in paths)


def test_edgar_parse_filed_date_extracts_iso_date(tmp_path):
    """SGML header line 'FILED AS OF DATE: 20240221' → '2024-02-21'."""
    from data.edgar import parse_filed_date

    sgml = (
        "<SEC-DOCUMENT>0001045810-24-000023.txt : 20240221\n"
        "<SEC-HEADER>\n"
        "ACCESSION NUMBER:		0001045810-24-000023\n"
        "CONFORMED SUBMISSION TYPE:	10-K\n"
        "PUBLIC DOCUMENT COUNT:		104\n"
        "CONFORMED PERIOD OF REPORT:	20240128\n"
        "FILED AS OF DATE:		20240221\n"
        "DATE AS OF CHANGE:		20240221\n"
    )
    path = tmp_path / "full-submission.txt"
    path.write_text(sgml)
    assert parse_filed_date(path) == "2024-02-21"


def test_edgar_parse_filed_date_returns_none_when_header_missing():
    from data.edgar import parse_filed_date

    nonexistent = Path("/tmp/this-file-does-not-exist-finaq.txt")
    assert parse_filed_date(nonexistent) is None


def test_edgar_parse_filed_date_returns_none_when_pattern_absent(tmp_path):
    from data.edgar import parse_filed_date

    path = tmp_path / "full-submission.txt"
    path.write_text("Random text with no SGML header at all\n" * 20)
    assert parse_filed_date(path) is None


# --- data/yfin.py ------------------------------------------------------------


def test_yfin_cache_hit_within_ttl(tmp_path, monkeypatch):
    from data import yfin

    monkeypatch.setattr(yfin, "CACHE_DIR", tmp_path)
    payload = {
        "price_history_5y": {},
        "income_stmt": {"a": 1},
        "balance_sheet": {},
        "cash_flow": {},
        "info": {"longName": "Stub"},
    }
    on_disk = {**payload, "_format_version": yfin.CACHE_FORMAT_VERSION}
    (tmp_path / "STUB.json").write_text(json.dumps(on_disk))

    # Cache file is fresh by default (just written).
    with patch.object(yfin, "_fetch_from_yfinance") as mock_fetch:
        result = yfin.get_financials("STUB")
    mock_fetch.assert_not_called()
    # Version field is internal — should be hidden from consumers.
    assert "_format_version" not in result
    assert result == payload


def test_yfin_cache_with_old_format_version_is_invalidated(tmp_path, monkeypatch):
    """A cache file from before the format bump must not be used silently."""
    from data import yfin

    monkeypatch.setattr(yfin, "CACHE_DIR", tmp_path)
    stale_payload = {
        "price_history_5y": {},
        "income_stmt": {"a": 1},
        "balance_sheet": {},
        "cash_flow": {},
        "info": {},
        "_format_version": yfin.CACHE_FORMAT_VERSION - 1,  # one version old
    }
    (tmp_path / "STUB.json").write_text(json.dumps(stale_payload))

    with patch.object(yfin, "_fetch_from_yfinance") as mock_fetch:
        mock_fetch.return_value = {k: {} for k in yfin.EXPECTED_KEYS}
        yfin.get_financials("STUB")
    mock_fetch.assert_called_once()


def test_yfin_cache_miss_after_ttl(tmp_path, monkeypatch):
    from data import yfin

    monkeypatch.setattr(yfin, "CACHE_DIR", tmp_path)
    cache_path = tmp_path / "STUB.json"
    cache_path.write_text(json.dumps({"price_history_5y": {}}))
    # Make the cache file look stale.
    stale_mtime = time.time() - (yfin.CACHE_TTL_SECONDS + 60)
    Path(cache_path).touch()
    import os

    os.utime(cache_path, (stale_mtime, stale_mtime))

    with patch.object(yfin, "_fetch_from_yfinance") as mock_fetch:
        mock_fetch.return_value = {k: {} for k in yfin.EXPECTED_KEYS}
        yfin.get_financials("STUB")
    mock_fetch.assert_called_once()


def test_yfin_returns_partial_dict_with_errors_field_on_failure(tmp_path, monkeypatch):
    from data import yfin

    monkeypatch.setattr(yfin, "CACHE_DIR", tmp_path)
    with patch.object(yfin, "_fetch_from_yfinance", side_effect=RuntimeError("kaboom")):
        result = yfin.get_financials("ZZZZ")
    assert "errors" in result
    assert all(k in result for k in yfin.EXPECTED_KEYS)


# --- data/chroma.py (pure-logic helpers, no real ChromaDB) -------------------


def test_chroma_split_into_items_extracts_codes_and_bodies():
    from data.chroma import _split_into_items

    text = (
        "Item 1A. Risk Factors\n"
        "Risks include macro and supply chain.\n"
        "Item 7. Management's Discussion and Analysis\n"
        "MD&A talks about revenue growth.\n"
        "Item 7A. Quantitative and Qualitative Disclosures\n"
        "Interest rate risk and currency exposure.\n"
    )
    items = _split_into_items(text)
    codes = [code for code, _, _ in items]
    assert codes == ["1A", "7", "7A"]
    assert "Risks include macro" in items[0][2]
    assert "MD&A talks" in items[1][2]


def test_chroma_split_into_items_returns_misc_when_no_headers():
    from data.chroma import _split_into_items

    text = "This document has no Item headers anywhere in it."
    items = _split_into_items(text)
    assert items == [("misc", "Unstructured", text)]


def test_chroma_chunk_tokens_respects_target_and_overlap():
    import tiktoken

    from data.chroma import CHUNK_OVERLAP_TOKENS, TARGET_CHUNK_TOKENS, _chunk_tokens

    encoder = tiktoken.get_encoding("cl100k_base")
    long_text = ("hello world " * 1000).strip()  # ~2000 tokens
    chunks = _chunk_tokens(long_text, encoder)
    assert len(chunks) >= 2

    sizes = [len(encoder.encode(c)) for c in chunks]
    # All chunks (except possibly the last) should be roughly the target size.
    assert all(s <= TARGET_CHUNK_TOKENS for s in sizes)
    assert sizes[0] >= TARGET_CHUNK_TOKENS - CHUNK_OVERLAP_TOKENS


def test_chroma_chunk_tokens_empty_text_yields_no_chunks():
    import tiktoken

    from data.chroma import _chunk_tokens

    encoder = tiktoken.get_encoding("cl100k_base")
    assert _chunk_tokens("", encoder) == []


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1A", "1A"),
        ("Item 1A", "1A"),
        ("ITEM 7A", "7A"),
        ("item 7", "7"),
        ("  Item 1A. Risk Factors  ", "1A"),
        ("7", "7"),
    ],
)
def test_chroma_normalize_item_filter_codes(raw, expected):
    from data.chroma import _normalize_item_filter

    assert _normalize_item_filter(raw) == expected


def test_chroma_filing_meta_from_path_extracts_kind_and_accession():
    from data.chroma import _filing_meta_from_path

    path = Path("data_cache/edgar/sec-edgar-filings/NVDA/10-K/0001-25-000123/full-submission.txt")
    kind, accession = _filing_meta_from_path(path)
    assert kind == "10-K"
    assert accession == "0001-25-000123"
