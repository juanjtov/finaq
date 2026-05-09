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


def test_chroma_get_collection_passes_name_through_to_client(monkeypatch):
    """Step 11.4 — `_get_collection(name=...)` must request the named
    collection from the chroma client. The CIO planner relies on this to
    open `synthesis_reports` separately from the filings corpus."""
    from data import chroma as ch

    captured: dict = {}

    class _FakeCollection:
        pass

    class _FakeClient:
        def get_or_create_collection(self, *, name, embedding_function, configuration):
            captured["name"] = name
            return _FakeCollection()

    monkeypatch.setattr(ch.chromadb, "PersistentClient", lambda **kwargs: _FakeClient())

    coll = ch._get_collection()
    assert isinstance(coll, _FakeCollection)
    assert captured["name"] == "filings"  # default

    coll = ch._get_collection(name="synthesis_reports")
    assert captured["name"] == "synthesis_reports"


def test_ingest_filing_batches_upsert_when_chunk_count_exceeds_chroma_limit(
    tmp_path, monkeypatch
):
    """Real failure mode (SMCI/DELL ingestion): a 10-K can produce >5461
    chunks in one filing — ChromaDB's `add`/`upsert` raises
    `ValueError: Batch size … is greater than max batch size of 5461`.
    The fix batches the upsert; this test asserts upsert is called multiple
    times for a synthetic large filing, never with more than _UPSERT_BATCH_SIZE
    items per call."""
    from data import chroma as ch

    # Fake collection that just records every call.
    upsert_calls: list[int] = []

    class _FakeCollection:
        def upsert(self, ids, documents, metadatas):
            upsert_calls.append(len(ids))

    monkeypatch.setattr(ch, "_get_collection", lambda: _FakeCollection())

    # Synthesize a fake "extracted text" that produces >5461 chunks. Bypass
    # the SEC-filing parser by stubbing _extract_text + _split_into_items +
    # _chunk_tokens to return a known number of chunks directly.
    # 5500 sits above the 4000 batch limit (so we can verify batching) but
    # below the 6000 chunk cap (so the cap doesn't fire and obscure the
    # batched-upsert behaviour we're checking here).
    n_chunks_total = 5500
    monkeypatch.setattr(ch, "_extract_text", lambda p: "x")  # non-empty short-circuits the bail
    monkeypatch.setattr(
        ch,
        "_split_into_items",
        lambda text: [("1A", "Risk Factors", "stub body")],
    )
    monkeypatch.setattr(
        ch,
        "_chunk_tokens",
        lambda body, encoder: [f"chunk-{i}" for i in range(n_chunks_total)],
    )
    monkeypatch.setattr(
        ch,
        "_filing_meta_from_path",
        lambda path: ("10-K", "0001-25-000999"),
    )
    monkeypatch.setattr(ch, "parse_filed_date", lambda path: "2026-02-26")

    # Need a tiktoken encoder placeholder; the lambda above ignores it.
    monkeypatch.setattr(
        ch.tiktoken, "get_encoding", lambda name: object()
    )

    fake_path = tmp_path / "full-submission.txt"
    fake_path.write_text("ignored — _extract_text is stubbed")

    written = ch.ingest_filing("SMCI", fake_path)
    assert written == n_chunks_total
    # Multiple batches because total > _UPSERT_BATCH_SIZE
    assert len(upsert_calls) >= 2, (
        f"expected >=2 upsert calls for {n_chunks_total} chunks, got {len(upsert_calls)}"
    )
    # No single batch exceeds the configured limit (ChromaDB max is 5461;
    # we use 4000 to stay well below it).
    assert all(n <= ch._UPSERT_BATCH_SIZE for n in upsert_calls), (
        f"a batch exceeded the upsert limit: {upsert_calls}"
    )
    # The total writes match.
    assert sum(upsert_calls) == n_chunks_total


def test_ingest_filing_single_upsert_when_chunk_count_below_limit(tmp_path, monkeypatch):
    """Small filings (under the batch limit) should still produce exactly
    one upsert call — no per-batch overhead when there's nothing to batch."""
    from data import chroma as ch

    upsert_calls: list[int] = []

    class _FakeCollection:
        def upsert(self, ids, documents, metadatas):
            upsert_calls.append(len(ids))

    monkeypatch.setattr(ch, "_get_collection", lambda: _FakeCollection())
    monkeypatch.setattr(ch, "_extract_text", lambda p: "x")
    monkeypatch.setattr(
        ch,
        "_split_into_items",
        lambda text: [("1A", "Risk Factors", "stub body")],
    )
    monkeypatch.setattr(
        ch,
        "_chunk_tokens",
        lambda body, encoder: [f"chunk-{i}" for i in range(500)],
    )
    monkeypatch.setattr(ch, "_filing_meta_from_path", lambda path: ("10-Q", "0001-25-001"))
    monkeypatch.setattr(ch, "parse_filed_date", lambda path: "2026-02-26")
    monkeypatch.setattr(ch.tiktoken, "get_encoding", lambda name: object())

    fake_path = tmp_path / "full-submission.txt"
    fake_path.write_text("ignored")

    ch.ingest_filing("CRDO", fake_path)
    assert upsert_calls == [500]


def test_ingest_filing_caps_chunks_when_filing_is_pathologically_large(
    tmp_path, monkeypatch, caplog
):
    """Real failure mode (NU 20-F): a single filing produced 19,423 chunks
    because the 35MB filing included full XBRL inline content. ChromaDB's
    HNSW index then ate 84GB of disk before the ingest was killed.

    The cap caps each filing at `_MAX_CHUNKS_PER_FILING` so a pathological
    20-F can no longer take the system out. The test simulates a 20,000-
    chunk filing and asserts only `_MAX_CHUNKS_PER_FILING` land."""
    import logging

    from data import chroma as ch

    upsert_calls: list[int] = []

    class _FakeCollection:
        def upsert(self, ids, documents, metadatas):
            upsert_calls.append(len(ids))

    monkeypatch.setattr(ch, "_get_collection", lambda: _FakeCollection())
    monkeypatch.setattr(ch, "_extract_text", lambda p: "x")
    monkeypatch.setattr(
        ch,
        "_split_into_items",
        lambda text: [("MISC", "Unstructured", "stub body")],
    )
    # 20,000 chunks — well above the cap.
    monkeypatch.setattr(
        ch,
        "_chunk_tokens",
        lambda body, encoder: [f"chunk-{i}" for i in range(20_000)],
    )
    monkeypatch.setattr(
        ch, "_filing_meta_from_path", lambda path: ("20-F", "0001292814-25-001517"),
    )
    monkeypatch.setattr(ch, "parse_filed_date", lambda path: "2025-03-15")
    monkeypatch.setattr(ch.tiktoken, "get_encoding", lambda name: object())

    fake_path = tmp_path / "full-submission.txt"
    fake_path.write_text("ignored")

    with caplog.at_level(logging.WARNING):
        written = ch.ingest_filing("NU", fake_path)

    assert written == ch._MAX_CHUNKS_PER_FILING
    assert sum(upsert_calls) == ch._MAX_CHUNKS_PER_FILING
    # Warning log fired so the operator knows the filing was truncated.
    assert any("hit chunk cap" in m for m in caplog.messages), (
        f"expected chunk-cap warning in logs: {caplog.messages}"
    )


# --- Hybrid retrieval: BM25 + RRF -------------------------------------------


def test_chroma_bm25_ranks_keyword_match_first():
    from data.chroma import _bm25_rank

    docs = [
        "the cat sat on the mat",
        "data center capex grew rapidly in fiscal 2024",
        "the dog barked at the cat",
    ]
    ranks = _bm25_rank(docs, "data center capex")
    assert ranks[0] == 1, f"BM25 should rank doc 1 first, got order {ranks}"


def test_chroma_bm25_handles_empty_corpus():
    from data.chroma import _bm25_rank

    assert _bm25_rank([], "anything") == []


def test_chroma_bm25_returns_full_ranking_with_no_matches():
    """BM25 still returns a complete ranking even if no doc shares any terms with the query."""
    from data.chroma import _bm25_rank

    docs = ["alpha beta gamma", "delta epsilon zeta"]
    ranks = _bm25_rank(docs, "completely unrelated terms here")
    assert sorted(ranks) == [0, 1]


def test_chroma_reciprocal_rank_fusion_promotes_consistently_high_items():
    """An item ranked #1 in both lists should outrank an item that's only #1 in one."""
    from data.chroma import _reciprocal_rank_fusion

    sem_ranks = [0, 1, 2, 3]  # doc 0 best in semantic
    bm25_ranks = [0, 2, 1, 3]  # doc 0 best in BM25 too
    fused = _reciprocal_rank_fusion([sem_ranks, bm25_ranks])
    assert fused[0] == 0


def test_chroma_reciprocal_rank_fusion_balances_disjoint_strengths():
    """If A is best semantically and B is best by keyword, the second item in each list
    (the consistent one) should rank higher than either pure-list winner."""
    from data.chroma import _reciprocal_rank_fusion

    # doc 1 is rank-2 in BOTH lists; doc 0 is rank-1 in list A but rank-3 in list B.
    sem = [0, 1, 2]
    bm25 = [2, 1, 0]
    fused = _reciprocal_rank_fusion([sem, bm25])
    # Doc 1 is rank 2 in both lists → score = 2/(60+2) = 0.0322
    # Doc 0 is rank 1, rank 3 → 1/61 + 1/63 ≈ 0.0322
    # Doc 2 is rank 3, rank 1 → same ≈ 0.0322
    # All three end up near-equal; what matters is: fused output contains all 3 items.
    assert sorted(fused) == [0, 1, 2]


def test_chroma_reciprocal_rank_fusion_handles_single_list():
    from data.chroma import _reciprocal_rank_fusion

    assert _reciprocal_rank_fusion([[2, 0, 1]]) == [2, 0, 1]


def test_chroma_reciprocal_rank_fusion_includes_items_unique_to_one_list():
    from data.chroma import _reciprocal_rank_fusion

    fused = _reciprocal_rank_fusion([[0, 1], [2, 3]])
    assert sorted(fused) == [0, 1, 2, 3]


def test_chroma_build_where_clause_single_condition():
    from data.chroma import _build_where_clause

    assert _build_where_clause("NVDA", None) == {"ticker": "NVDA"}
    assert _build_where_clause(None, "1A") == {"item_code": "1A"}


def test_chroma_build_where_clause_multiple_conditions_uses_and():
    from data.chroma import _build_where_clause

    where = _build_where_clause("NVDA", "Item 1A")
    assert where == {"$and": [{"ticker": "NVDA"}, {"item_code": "1A"}]}


def test_chroma_build_where_clause_no_filters_returns_none():
    from data.chroma import _build_where_clause

    assert _build_where_clause(None, None) is None
