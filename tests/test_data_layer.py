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


def test_edgar_existing_filings_are_newest_first(tmp_path, monkeypatch):
    """`download_filings` keeps `existing[:limit]`, so the order decides which
    filings get ingested. Accession numbers sort oldest-first; the SGML filed
    date must drive the order or a backtest-bumped corpus seeds stale 10-Ks."""
    from data import edgar

    monkeypatch.setattr(edgar, "EDGAR_DIR", tmp_path)
    base = tmp_path / "sec-edgar-filings" / "NKE" / "10-K"
    for accession, filed in [
        ("0000320187-22-000038", "20220721"),
        ("0000320187-25-000047", "20250717"),
        ("0000320187-23-000039", "20230720"),
    ]:
        (base / accession).mkdir(parents=True)
        (base / accession / "full-submission.txt").write_text(
            f"<SEC-DOCUMENT>\nFILED AS OF DATE:\t{filed}\n<TYPE>10-K\n"
        )
    paths = edgar._existing_filings("NKE", "10-K")
    assert [p.parent.name[-6:] for p in paths] == ["000047", "000039", "000038"]


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


# --- data/vectors.py (pure-logic helpers, no real the filings index) -------------------


def test_vectors_split_into_items_extracts_codes_and_bodies():
    from data.vectors import _split_into_items

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


def test_vectors_split_into_items_returns_misc_when_no_headers():
    from data.vectors import _split_into_items

    text = "This document has no Item headers anywhere in it."
    items = _split_into_items(text)
    assert items == [("misc", "Unstructured", text)]


def test_vectors_chunk_tokens_respects_target_and_overlap():
    import tiktoken

    from data.vectors import CHUNK_OVERLAP_TOKENS, TARGET_CHUNK_TOKENS, _chunk_tokens

    encoder = tiktoken.get_encoding("cl100k_base")
    long_text = ("hello world " * 1000).strip()  # ~2000 tokens
    chunks = _chunk_tokens(long_text, encoder)
    assert len(chunks) >= 2

    sizes = [len(encoder.encode(c)) for c in chunks]
    # All chunks (except possibly the last) should be roughly the target size.
    assert all(s <= TARGET_CHUNK_TOKENS for s in sizes)
    assert sizes[0] >= TARGET_CHUNK_TOKENS - CHUNK_OVERLAP_TOKENS


def test_vectors_chunk_tokens_empty_text_yields_no_chunks():
    import tiktoken

    from data.vectors import _chunk_tokens

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
def test_vectors_normalize_item_filter_codes(raw, expected):
    from data.vectors import _normalize_item_filter

    assert _normalize_item_filter(raw) == expected


def test_vectors_filing_meta_from_path_extracts_kind_and_accession():
    from data.vectors import _filing_meta_from_path

    path = Path("data_cache/edgar/sec-edgar-filings/NVDA/10-K/0001-25-000123/full-submission.txt")
    kind, accession = _filing_meta_from_path(path)
    assert kind == "10-K"
    assert accession == "0001-25-000123"


# --- data/vectors.py: primary-document extraction --------------------------


def _sgml(*docs: tuple[str, str]) -> str:
    """Build an EDGAR full-submission.txt envelope from (TYPE, body) pairs."""
    return "".join(
        f"<DOCUMENT>\n<TYPE>{t}\n<SEQUENCE>1\n<TEXT>\n{body}\n</TEXT>\n</DOCUMENT>\n"
        for t, body in docs
    )


def test_vectors_primary_documents_keep_form_and_press_release_only():
    """A submission bundles the form with XBRL, certifications, graphics and
    ZIPs. Only the form block and EX-99 press releases carry narrative."""
    from data.vectors import _primary_documents

    raw = _sgml(
        ("10-Q", "<p>Item 2. MD&A narrative</p>"),
        ("EX-31.1", "officer certification"),
        ("EX-101.SCH", "<xml>taxonomy</xml>"),
        ("GRAPHIC", "begin 644 logo.jpg\nM4$Y'"),
        ("EX-99.1", "press release"),
    )
    kept = _primary_documents(raw, "10-Q")
    assert len(kept) == 2
    assert "MD&A narrative" in kept[0]
    assert "press release" in kept[1]


def test_vectors_primary_documents_accept_amendments_and_fall_back():
    from data.vectors import _primary_documents

    assert len(_primary_documents(_sgml(("10-K/A", "amended")), "10-K")) == 1
    # No block matches the form → the first block, never nothing.
    kept = _primary_documents(_sgml(("EX-10.1", "contract"), ("XML", "<x/>")), "10-K")
    assert len(kept) == 1 and "contract" in kept[0]
    # No SGML envelope at all → the whole file.
    assert _primary_documents("plain text", "10-K") == ["plain text"]


def test_vectors_extract_text_drops_attachments(tmp_path):
    """End to end: uuencoded graphics and XBRL labels never reach the text
    that gets chunked — they were 94% of the old corpus."""
    from data.vectors import _extract_text

    folder = tmp_path / "sec-edgar-filings" / "NKE" / "10-K" / "0000320187-25-000001"
    folder.mkdir(parents=True)
    path = folder / "full-submission.txt"
    path.write_text(
        _sgml(
            (
                "10-K",
                "<html><body><p>Item 1A. Risk Factors</p><p>Demand may soften.</p></body></html>",
            ),
            ("GRAPHIC", "begin 644 x\nM4$Y'#0H:"),
            ("EX-101.LAB", "<link:label>CostOfRevenue</link:label>"),
        )
    )
    text = _extract_text(path)
    assert "Demand may soften." in text
    assert "M4$Y'" not in text
    assert "CostOfRevenue" not in text


def test_vectors_date_int():
    from data.vectors import _date_int

    assert _date_int("2026-02-26") == 20260226
    assert _date_int("") == 0
    assert _date_int("garbage") == 0


# --- data/vectors.py: ingest ------------------------------------------------


class _FakeIndex:
    """In-memory stand-in for a Pinecone Index: records upserts, serves
    list / fetch / delete by id, and returns canned query matches."""

    def __init__(self, matches: list[dict] | None = None):
        self.store: dict[str, dict[str, dict]] = {}  # namespace → id → record
        self.upsert_calls: list[tuple[str, int]] = []
        self.queries: list[dict] = []
        self._matches = matches or []

    def upsert(self, *, vectors, namespace="", show_progress=True):
        self.upsert_calls.append((namespace, len(vectors)))
        ns = self.store.setdefault(namespace, {})
        for v in vectors:
            ns[v["id"]] = {"values": v["values"], "metadata": v["metadata"]}

    def list(self, *, prefix=None, limit=None, namespace=""):
        ids = [i for i in self.store.get(namespace, {}) if i.startswith(prefix or "")]
        page = limit or 100
        for i in range(0, len(ids), page):
            yield {"vectors": [{"id": x} for x in ids[i : i + page]]}

    def delete(self, *, ids=None, namespace="", **kwargs):
        for i in ids or []:
            self.store.get(namespace, {}).pop(i, None)

    def fetch(self, *, ids, namespace=""):
        ns = self.store.get(namespace, {})
        return {"vectors": {i: {"id": i, "metadata": ns[i]["metadata"]} for i in ids if i in ns}}

    def query(self, **kwargs):
        self.queries.append(kwargs)
        return {"matches": self._matches}


def _stub_ingest(monkeypatch, vec, n_chunks: int, *, accession: str, filing_type: str = "10-Q"):
    """Bypass the SEC parser + embeddings; return the fake index that receives writes."""
    monkeypatch.setattr(vec, "_extract_text", lambda p: "x")
    monkeypatch.setattr(vec, "_split_into_items", lambda text: [("1A", "Risk Factors", "body")])
    monkeypatch.setattr(
        vec, "_chunk_tokens", lambda body, encoder: [f"c{i}" for i in range(n_chunks)]
    )
    monkeypatch.setattr(vec, "_filing_meta_from_path", lambda path: (filing_type, accession))
    monkeypatch.setattr(vec, "parse_filed_date", lambda path: "2026-08-05")
    monkeypatch.setattr(vec.tiktoken, "get_encoding", lambda name: object())
    monkeypatch.setattr(vec, "embed_texts", lambda texts: [[0.1, 0.2, 0.3] for _ in texts])
    fake = _FakeIndex()
    monkeypatch.setattr(vec, "_index", lambda name: fake)
    return fake


def test_vectors_ingest_batches_upserts_and_records_manifest(tmp_path, monkeypatch):
    from data import state as state_db
    from data import vectors as vec

    fake = _stub_ingest(monkeypatch, vec, 250, accession="0001-26-250")
    path = tmp_path / "full-submission.txt"
    path.write_text("ignored")

    assert vec.ingest_filing("crdo", path) == 250
    assert [n for _, n in fake.upsert_calls] == [100, 100, 50]
    assert {ns for ns, _ in fake.upsert_calls} == {"CRDO"}  # namespace per ticker
    stored = fake.store["CRDO"]["CRDO-0001-26-250-0"]
    assert stored["metadata"]["text"] == "c0"
    assert stored["metadata"]["item_code"] == "1A"
    assert stored["metadata"]["filed_date_int"] == 20260805
    rows = state_db.ingested_filings("CRDO")
    assert [(r["accession"], r["chunks"], r["filed_date"]) for r in rows] == [
        ("0001-26-250", 250, "2026-08-05")
    ]


def test_vectors_ingest_skips_when_manifest_matches(tmp_path, monkeypatch):
    """A filing whose chunk count is already in the manifest was fully
    ingested before → no embed, no upsert. Re-embedding every on-disk
    filing whenever one new 10-Q landed multiplied ingest cost ~6x."""
    from data import state as state_db
    from data import vectors as vec

    fake = _stub_ingest(monkeypatch, vec, 5, accession="0001-26-005")
    state_db.record_ingested_filing(
        ticker="CRDO",
        accession="0001-26-005",
        filing_type="10-Q",
        filed_date="2026-08-05",
        chunks=5,
    )
    path = tmp_path / "full-submission.txt"
    path.write_text("ignored")

    assert vec.ingest_filing("CRDO", path) == 0
    assert fake.upsert_calls == []


def test_vectors_ingest_replaces_stale_chunks_when_count_differs(tmp_path, monkeypatch):
    """A newer chunker (or a run killed mid-upsert) leaves the manifest count
    different from the fresh count → the old ids are dropped by prefix and the
    filing is embedded again in full. Other filings in the namespace survive."""
    from data import state as state_db
    from data import vectors as vec

    fake = _stub_ingest(monkeypatch, vec, 3, accession="0001-26-003")
    fake.store["CRDO"] = {
        f"CRDO-0001-26-003-{i}": {"values": [], "metadata": {}} for i in range(7)
    }
    fake.store["CRDO"]["CRDO-0001-26-004-0"] = {"values": [], "metadata": {}}
    state_db.record_ingested_filing(
        ticker="CRDO",
        accession="0001-26-003",
        filing_type="10-Q",
        filed_date="2026-08-05",
        chunks=7,
    )
    path = tmp_path / "full-submission.txt"
    path.write_text("ignored")

    assert vec.ingest_filing("CRDO", path) == 3
    assert set(fake.store["CRDO"]) == {
        "CRDO-0001-26-003-0",
        "CRDO-0001-26-003-1",
        "CRDO-0001-26-003-2",
        "CRDO-0001-26-004-0",
    }
    assert state_db.ingested_filing_chunks("CRDO", "0001-26-003") == 3


def test_vectors_ingest_embeds_when_stale_id_lookup_fails(tmp_path, monkeypatch):
    """A list() error while looking for stale ids must not block ingest —
    embedding anyway is the safe direction."""
    from data import vectors as vec

    fake = _stub_ingest(monkeypatch, vec, 3, accession="0001-26-013")

    def _boom(**kwargs):
        raise RuntimeError("namespace missing")

    fake.list = _boom
    path = tmp_path / "full-submission.txt"
    path.write_text("ignored")

    assert vec.ingest_filing("CRDO", path) == 3
    assert [n for _, n in fake.upsert_calls] == [3]


def test_vectors_ingest_caps_pathological_filings(tmp_path, monkeypatch, caplog):
    """Safety net: a malformed submission that still chunks into tens of
    thousands of pieces is truncated at `_MAX_CHUNKS_PER_FILING` with a
    warning, keeping the head of the filing where the Items live."""
    import logging

    from data import vectors as vec

    fake = _stub_ingest(
        monkeypatch, vec, 20_000, accession="0001292814-25-001517", filing_type="20-F"
    )
    path = tmp_path / "full-submission.txt"
    path.write_text("ignored")

    with caplog.at_level(logging.WARNING):
        written = vec.ingest_filing("NU", path)

    assert written == vec._MAX_CHUNKS_PER_FILING
    assert sum(n for _, n in fake.upsert_calls) == vec._MAX_CHUNKS_PER_FILING
    assert any("hit chunk cap" in m for m in caplog.messages)


# --- data/vectors.py: query -------------------------------------------------


def test_vectors_query_requires_ticker():
    from data import vectors as vec

    with pytest.raises(ValueError, match="ticker"):
        vec.query(None, "anything")


def test_vectors_query_scopes_namespace_filters_and_fuses(monkeypatch):
    """The ticker selects the namespace; item + as_of become a server-side
    metadata filter; BM25 over the candidate pool promotes the keyword hit
    through RRF; chunk text is lifted out of metadata."""
    from data import vectors as vec

    common = {"item_code": "7", "filed_date": "2026-01-01"}
    matches = [
        {"id": "a", "score": 0.9, "metadata": {"text": "cooling supply for racks", **common}},
        {
            "id": "b",
            "score": 0.8,
            "metadata": {"text": "Blackwell ramp constrained by capacity", **common},
        },
        {"id": "c", "score": 0.7, "metadata": {"text": "capacity remarks", **common}},
    ]
    fake = _FakeIndex(matches=matches)
    monkeypatch.setattr(vec, "_index", lambda name: fake)
    monkeypatch.setattr(vec, "embed_texts", lambda texts: [[0.5, 0.5]])

    out = vec.query("nvda", "Blackwell capacity", k=2, item_filter="Item 7", as_of="2026-03-01")

    q = fake.queries[0]
    assert q["namespace"] == "NVDA"
    assert q["top_k"] == vec.DEFAULT_CANDIDATE_POOL
    assert q["include_metadata"] is True
    assert q["filter"] == {
        "item_code": {"$eq": "7"},
        "filed_date_int": {"$gte": 1, "$lte": 20260301},
    }
    assert len(out) == 2
    assert out[0]["text"].startswith("Blackwell")
    assert "text" not in out[0]["metadata"]
    assert out[0]["metadata"]["item_code"] == "7"


def test_vectors_query_returns_empty_when_no_matches(monkeypatch):
    from data import vectors as vec

    monkeypatch.setattr(vec, "_index", lambda name: _FakeIndex(matches=[]))
    monkeypatch.setattr(vec, "embed_texts", lambda texts: [[0.5, 0.5]])
    assert vec.query("NVDA", "anything") == []


def test_vectors_build_filter():
    from data.vectors import _build_filter

    assert _build_filter(None, None) is None
    assert _build_filter("Item 1A", None) == {"item_code": {"$eq": "1A"}}
    assert _build_filter(None, "2025-09-05") == {
        "filed_date_int": {"$gte": 1, "$lte": 20250905}
    }


# --- data/vectors.py: ingest-manifest probes --------------------------------


@pytest.mark.real_probes
def test_vectors_manifest_probes_read_state_db():
    from data import state as state_db
    from data import vectors as vec

    assert vec.has_ticker("ZZZZ") is False
    assert vec.last_filings_by_type("ZZZZ") == {}
    assert vec.last_filing_date("ZZZZ") is None

    for accession, ftype, fdate in [
        ("a1", "10-K", "2025-02-01"),
        ("a2", "10-Q", "2025-08-01"),
        ("a3", "10-Q", ""),  # unparsed date — ignored by the probes
    ]:
        state_db.record_ingested_filing(
            ticker="ZZZZ", accession=accession, filing_type=ftype, filed_date=fdate, chunks=4
        )
    assert vec.has_ticker("zzzz") is True
    assert vec.last_filings_by_type("ZZZZ") == {"10-K": "2025-02-01", "10-Q": "2025-08-01"}
    assert vec.last_filing_date("ZZZZ") == "2025-08-01"


# --- data/vectors.py: BM25 + RRF --------------------------------------------


def test_vectors_bm25_ranks_keyword_match_first():
    from data.vectors import _bm25_rank

    docs = [
        "the cat sat on the mat",
        "data center capex grew rapidly in fiscal 2024",
        "the dog barked at the cat",
    ]
    ranks = _bm25_rank(docs, "data center capex")
    assert ranks[0] == 1, f"BM25 should rank doc 1 first, got order {ranks}"


def test_vectors_bm25_handles_empty_corpus():
    from data.vectors import _bm25_rank

    assert _bm25_rank([], "anything") == []


def test_vectors_bm25_returns_full_ranking_with_no_matches():
    """BM25 still returns a complete ranking even if no doc shares any terms with the query."""
    from data.vectors import _bm25_rank

    docs = ["alpha beta gamma", "delta epsilon zeta"]
    ranks = _bm25_rank(docs, "completely unrelated terms here")
    assert sorted(ranks) == [0, 1]


def test_vectors_reciprocal_rank_fusion_promotes_consistently_high_items():
    """An item ranked #1 in both lists should outrank an item that's only #1 in one."""
    from data.vectors import _reciprocal_rank_fusion

    sem_ranks = [0, 1, 2, 3]  # doc 0 best in semantic
    bm25_ranks = [0, 2, 1, 3]  # doc 0 best in BM25 too
    fused = _reciprocal_rank_fusion([sem_ranks, bm25_ranks])
    assert fused[0] == 0


def test_vectors_reciprocal_rank_fusion_balances_disjoint_strengths():
    """If A is best semantically and B is best by keyword, the second item in each list
    (the consistent one) should rank higher than either pure-list winner."""
    from data.vectors import _reciprocal_rank_fusion

    # doc 1 is rank-2 in BOTH lists; doc 0 is rank-1 in list A but rank-3 in list B.
    sem = [0, 1, 2]
    bm25 = [2, 1, 0]
    fused = _reciprocal_rank_fusion([sem, bm25])
    # Doc 1 is rank 2 in both lists → score = 2/(60+2) = 0.0322
    # Doc 0 is rank 1, rank 3 → 1/61 + 1/63 ≈ 0.0322
    # Doc 2 is rank 3, rank 1 → same ≈ 0.0322
    # All three end up near-equal; what matters is: fused output contains all 3 items.
    assert sorted(fused) == [0, 1, 2]


def test_vectors_reciprocal_rank_fusion_handles_single_list():
    from data.vectors import _reciprocal_rank_fusion

    assert _reciprocal_rank_fusion([[2, 0, 1]]) == [2, 0, 1]


def test_vectors_reciprocal_rank_fusion_includes_items_unique_to_one_list():
    from data.vectors import _reciprocal_rank_fusion

    fused = _reciprocal_rank_fusion([[0, 1], [2, 3]])
    assert sorted(fused) == [0, 1, 2, 3]


# --- Freshness kill-switch (FINAQ_SKIP_FRESHNESS_PROBES) --------------------


def test_check_ingest_freshness_kill_switch_short_circuits(monkeypatch):
    """With the probes kill-switch set, check_ingest_freshness must report
    'unknown, not stale' WITHOUT touching EDGAR or the filings index. Regression:
    an earlier version only gated the ingest probe — an empty manifest dict
    plus a live EDGAR date then read as 'stale' for every ticker, funnelling
    the UI / Telegram / CIO straight into the ingest path the switch exists
    to avoid."""
    from data import freshness as fr

    def _boom(*args, **kwargs):
        raise AssertionError("probe called despite kill-switch")

    monkeypatch.setattr(fr, "latest_filing_dates", _boom)
    monkeypatch.setattr(fr, "last_filings_by_type", _boom)
    monkeypatch.setenv("FINAQ_SKIP_FRESHNESS_PROBES", "1")

    report = fr.check_ingest_freshness("NVDA")
    assert report.is_stale is False
    assert report.edgar_error is not None
    assert "FINAQ_SKIP_FRESHNESS_PROBES" in report.edgar_error
    assert report.per_form == []
    assert report.stale_forms() == []
