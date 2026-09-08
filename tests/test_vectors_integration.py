"""Live Pinecone + OpenRouter round-trip for `data/vectors.py`.

`pytest -m integration tests/test_vectors_integration.py`

Ingests a synthetic two-document SGML filing under a throwaway ticker
namespace, queries it back with an item filter and a backtest cutoff, then
deletes the namespace so the index stays clean. Serverless indexes are
eventually consistent, so writes are polled before querying.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

TICKER = "ZZTEST"
ACCESSION = "0000000000-26-000001"
REPORT_RUN_ID = f"{TICKER}__itest__abc123"

_SUBMISSION = f"""<SEC-DOCUMENT>0000000000-26-000001.txt : 20260805
<SEC-HEADER>0000000000-26-000001.hdr.sgml : 20260805
ACCESSION NUMBER:\t\t{ACCESSION}
CONFORMED SUBMISSION TYPE:\t10-Q
FILED AS OF DATE:\t\t20260805
</SEC-HEADER>
<DOCUMENT>
<TYPE>10-Q
<SEQUENCE>1
<TEXT>
<html><body>
<p>Item 1A. Risk Factors</p>
<p>Sneaker demand in Greater China may soften if wholesale partners cut orders.</p>
<p>Item 2. Management's Discussion and Analysis</p>
<p>Revenue grew nine percent on direct-to-consumer strength.</p>
</body></html>
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>GRAPHIC
<SEQUENCE>2
<TEXT>
begin 644 logo.jpg
M4$Y'#0H:"@````-24A$4@```!`````0"`8```#_
end
</TEXT>
</DOCUMENT>
</SEC-DOCUMENT>
"""


def _namespace_count(index, namespace: str) -> int:
    stats = index.describe_index_stats()
    namespaces = getattr(stats, "namespaces", None) or {}
    entry = namespaces.get(namespace)
    if entry is None:
        return 0
    count = getattr(entry, "vector_count", None)
    return int(count if count is not None else entry.get("vector_count", 0))


def _wait_for_count(index, namespace: str, expected: int, timeout_s: float = 90) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _namespace_count(index, namespace) == expected:
            return
        time.sleep(3)
    raise AssertionError(f"{namespace}: expected {expected} vectors, got {_namespace_count(index, namespace)}")


@pytest.fixture
def synthetic_filing(tmp_path: Path) -> Path:
    folder = tmp_path / "sec-edgar-filings" / TICKER / "10-Q" / ACCESSION
    folder.mkdir(parents=True)
    path = folder / "full-submission.txt"
    path.write_text(_SUBMISSION)
    return path


def test_filings_ingest_query_round_trip(synthetic_filing: Path):
    from data.vectors import INDEX_FILINGS, _index, has_ticker, ingest_filing, query

    index = _index(INDEX_FILINGS)
    try:
        n = ingest_filing(TICKER, synthetic_filing)
        assert 0 < n < 10, n  # the GRAPHIC block must not add chunks
        _wait_for_count(index, TICKER, n)
        assert has_ticker(TICKER)

        risk = query(TICKER, "Greater China demand risk", k=3, item_filter="1A")
        assert risk, "expected a Risk Factors chunk"
        assert all(r["metadata"]["item_code"] == "1A" for r in risk)
        assert "Greater China" in risk[0]["text"]
        assert risk[0]["metadata"]["filed_date"] == "2026-08-05"
        assert "text" not in risk[0]["metadata"]

        # Backtest cutoff before the filing date → nothing; after → hits.
        assert query(TICKER, "revenue growth", k=3, as_of="2026-01-01") == []
        assert query(TICKER, "revenue growth", k=3, as_of="2026-12-31")

        # Second run is a no-op (manifest count matches).
        assert ingest_filing(TICKER, synthetic_filing) == 0
    finally:
        index.delete(delete_all=True, namespace=TICKER)


def test_reports_upsert_fetch_query_round_trip():
    from data.vectors import (
        INDEX_REPORTS,
        REPORTS_NAMESPACE,
        _index,
        _list_ids,
        fetch_reports,
        query_reports,
        upsert_reports,
    )

    index = _index(INDEX_REPORTS)
    ids = [f"{REPORT_RUN_ID}-watchlist", f"{REPORT_RUN_ID}-bull_case"]
    docs = [
        "Q3 earnings call — listen for Greater China wholesale order trends (news)",
        "Direct-to-consumer mix keeps expanding gross margin",
    ]
    metas = [
        {"run_id": REPORT_RUN_ID, "ticker": TICKER, "thesis": "itest", "section": "Watchlist", "date": "2026-08-01"},
        {"run_id": REPORT_RUN_ID, "ticker": TICKER, "thesis": "itest", "section": "Bull case", "date": "2026-08-01"},
    ]
    try:
        assert upsert_reports(ids, docs, metas) == 2
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if len(_list_ids(index, prefix=f"{TICKER}__", namespace=REPORTS_NAMESPACE)) == 2:
                break
            time.sleep(3)

        watch = fetch_reports(TICKER, thesis="itest", section="Watchlist")
        assert len(watch) == 1
        assert watch[0]["text"].startswith("Q3 earnings call")
        assert watch[0]["metadata"]["run_id"] == REPORT_RUN_ID

        hits = query_reports(
            "gross margin from direct to consumer",
            where={"$and": [{"ticker": TICKER}, {"thesis": "itest"}]},
            top_k=2,
        )
        assert hits and hits[0]["metadata"]["section"] == "Bull case"
    finally:
        index.delete(ids=ids, namespace=REPORTS_NAMESPACE)
