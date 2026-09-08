"""Pinecone-backed vector store for FINAQ's two RAG corpora.

Two serverless indexes, both cosine over the env-configured OpenRouter
embedding model (`MODEL_EMBEDDINGS`):

  * `PINECONE_INDEX_FILINGS` (default `finaq-filings`) — SEC filing chunks.
    One namespace per ticker, so every query is scoped to a few hundred
    vectors and a ticker can be re-ingested by wiping its namespace.
    Metadata: {ticker, filing_type, accession, filed_date, filed_date_int,
    item_code, item_label, text}.
  * `PINECONE_INDEX_REPORTS` (default `finaq-reports`) — sections of past
    Synthesis reports for the CIO planner (`cio/rag.py`). Single namespace;
    ids are `{TICKER}__{thesis}__{hash}-{section}` so one ticker's chunks
    can be listed by id prefix.

Ingest bookkeeping (which accessions are indexed, with how many chunks)
lives in SQLite (`data/state.py::ingested_filings`). The freshness gate
probes dozens of tickers per CIO cycle and must not pay a network
round-trip per probe.

Chunking parses only the primary document of each EDGAR submission — the
10-K / 10-Q / 20-F / 6-K block, plus EX-99 press-release exhibits. The
rest of the submission (XBRL taxonomy XML, contract exhibits, officer
certifications, uuencoded graphics and ZIPs) is skipped; before this rule
94% of the corpus was attachment noise labelled as the filing's last Item.

Retrieval is hybrid:
  1. Metadata filter (item_code, and filed_date_int ≤ as_of for backtests)
     applied server-side before similarity.
  2. Semantic search returns the top `candidate_pool` chunks.
  3. BM25 keyword search runs over the same pool.
  4. Reciprocal Rank Fusion merges both rankings into the final top-k.

Cross-encoder re-ranking is intentionally NOT included; see docs/POSTPONED.md §2.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import tiktoken
from bs4 import BeautifulSoup
from pinecone import Pinecone, ServerlessSpec
from rank_bm25 import BM25Okapi

from data import state as state_db
from data.edgar import parse_filed_date
from utils import logger, tenacity_retry
from utils.models import MODEL_EMBEDDINGS
from utils.openrouter import get_client

INDEX_FILINGS = os.getenv("PINECONE_INDEX_FILINGS", "finaq-filings")
INDEX_REPORTS = os.getenv("PINECONE_INDEX_REPORTS", "finaq-reports")
REPORTS_NAMESPACE = "reports"
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")  # the Starter tier's region
DISTANCE_METRIC = "cosine"
TARGET_CHUNK_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 100
EMBED_BATCH_SIZE = 100  # OpenAI embeddings API batch limit
UPSERT_BATCH_SIZE = 100  # ~1MB of vectors + text per request; Pinecone caps requests at 2MB
ID_BATCH_SIZE = 100  # list() page size; also used for fetch / delete-by-id batches
TOKENIZER = "cl100k_base"
DEFAULT_CANDIDATE_POOL = 60  # semantic top-N, pre-fusion
RRF_K = 60  # standard reciprocal-rank-fusion constant
PRIMARY_EXHIBIT_PREFIX = "EX-99"  # press releases attached to 6-K / 8-K submissions

# Hard cap on chunks per filing — a safety net against a pathological
# submission. With primary-document extraction a typical 10-K runs a few
# hundred chunks, so this only fires on something malformed.
_MAX_CHUNKS_PER_FILING = 6000

# Lightweight English stopword list for BM25 tokenisation. Removing high-frequency
# function words sharpens the IDF signal so rare, discriminative terms (e.g.,
# "Blackwell", "capex", "constraint") drive ranking instead of "the"/"of"/"and".
# Curated subset of the canonical NLTK English stopword list (~150 entries).
BM25_STOPWORDS = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren't",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "can't",
        "cannot",
        "could",
        "couldn't",
        "did",
        "didn't",
        "do",
        "does",
        "doesn't",
        "doing",
        "don",
        "don't",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "hadn't",
        "has",
        "hasn't",
        "have",
        "haven't",
        "having",
        "he",
        "he'd",
        "he'll",
        "he's",
        "her",
        "here",
        "here's",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "how's",
        "i",
        "i'd",
        "i'll",
        "i'm",
        "i've",
        "if",
        "in",
        "into",
        "is",
        "isn't",
        "it",
        "it's",
        "its",
        "itself",
        "just",
        "let's",
        "me",
        "more",
        "most",
        "mustn't",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "shan't",
        "she",
        "she'd",
        "she'll",
        "she's",
        "should",
        "shouldn't",
        "so",
        "some",
        "such",
        "than",
        "that",
        "that's",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "there's",
        "these",
        "they",
        "they'd",
        "they'll",
        "they're",
        "they've",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "wasn't",
        "we",
        "we'd",
        "we'll",
        "we're",
        "we've",
        "were",
        "weren't",
        "what",
        "what's",
        "when",
        "when's",
        "where",
        "where's",
        "which",
        "while",
        "who",
        "who's",
        "whom",
        "why",
        "why's",
        "with",
        "won't",
        "would",
        "wouldn't",
        "you",
        "you'd",
        "you'll",
        "you're",
        "you've",
        "your",
        "yours",
        "yourself",
        "yourselves",
    ]
)
_WORD_RE = re.compile(r"[a-z0-9]+")

# Item header in 10-K / 10-Q: "Item 1A. Risk Factors", "ITEM 7.", etc.
ITEM_HEADER_RE = re.compile(
    r"^\s*item\s+(\d{1,2}[A-Z]?)\.?\s*(.{0,120}?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# SGML envelope of an EDGAR full-submission.txt: one <DOCUMENT> block per
# attached file, each announcing its <TYPE> (10-Q, EX-101.SCH, GRAPHIC, …).
_DOCUMENT_RE = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.DOTALL)
_TYPE_RE = re.compile(r"<TYPE>([^\s<]+)")
_TEXT_RE = re.compile(r"<TEXT>(.*?)</TEXT>", re.DOTALL)


# --- Embeddings ------------------------------------------------------------


@tenacity_retry
def _embed_batch(batch: list[str]) -> list[list[float]]:
    resp = get_client().embeddings.create(model=MODEL_EMBEDDINGS, input=batch)
    return [d.embedding for d in resp.data]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed via OpenRouter in API-sized batches. Ingest and query share it so
    both sides of the cosine live in the same space."""
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        out.extend(_embed_batch(list(texts[i : i + EMBED_BATCH_SIZE])))
    return out


# --- Pinecone client -------------------------------------------------------


@lru_cache(maxsize=1)
def _client() -> Pinecone:
    key = os.getenv("PINECONE_API_KEY")
    if not key:
        raise RuntimeError("PINECONE_API_KEY is not set — add it to .env")
    return Pinecone(api_key=key)


@lru_cache(maxsize=1)
def _embedding_dim() -> int:
    """Width of the configured embedding model, probed once per process so the
    index schema follows MODEL_EMBEDDINGS instead of a hardcoded number."""
    return len(embed_texts(["dimension probe"])[0])


@lru_cache(maxsize=4)
def _index(name: str):
    """Open a serverless index, creating it on first use (`create_index`
    blocks until the index is ready). Cached per process; tests monkeypatch
    this seam with an in-memory fake."""
    pc = _client()
    if not pc.has_index(name):
        logger.info(
            f"[vectors] creating Pinecone index {name} ({PINECONE_CLOUD}/{PINECONE_REGION})"
        )
        pc.create_index(
            name=name,
            dimension=_embedding_dim(),
            metric=DISTANCE_METRIC,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    return pc.Index(name)


def _get(obj, key: str, default=None):
    """Read `key` from an SDK response object or a plain dict (tests)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


@tenacity_retry
def _query_index(index, **kwargs):
    return index.query(include_metadata=True, **kwargs)


def _list_ids(index, *, prefix: str, namespace: str) -> list[str]:
    """Every vector id in `namespace` starting with `prefix`."""
    ids: list[str] = []
    for page in index.list(prefix=prefix, limit=ID_BATCH_SIZE, namespace=namespace):
        for item in _get(page, "vectors", None) or page:
            ids.append(item if isinstance(item, str) else _get(item, "id"))
    return ids


def _delete_by_prefix(index, prefix: str, *, namespace: str) -> int:
    """Delete every vector whose id starts with `prefix`. Serverless indexes
    can't delete by metadata filter, so this lists ids then deletes by id.
    A missing namespace is not an error — there is simply nothing to drop."""
    try:
        ids = _list_ids(index, prefix=prefix, namespace=namespace)
    except Exception as e:
        logger.warning(f"[vectors] list {namespace}/{prefix}* failed: {e} — not deleting")
        return 0
    for i in range(0, len(ids), ID_BATCH_SIZE):
        index.delete(ids=ids[i : i + ID_BATCH_SIZE], namespace=namespace)
    return len(ids)


def _upsert(index, *, namespace: str, ids: list[str], docs: list[str], metas: list[dict]) -> None:
    """Embed + upsert in request-sized batches. The chunk text rides along in
    metadata so retrieval can return it without a second store."""
    for i in range(0, len(ids), UPSERT_BATCH_SIZE):
        sl = slice(i, i + UPSERT_BATCH_SIZE)
        vectors = embed_texts(docs[sl])
        index.upsert(
            vectors=[
                {"id": _id, "values": vec, "metadata": {**meta, "text": doc}}
                for _id, vec, meta, doc in zip(ids[sl], vectors, metas[sl], docs[sl], strict=True)
            ],
            namespace=namespace,
            show_progress=False,
        )


def _unpack_matches(response) -> list[dict]:
    """Query response → `[{text, metadata, score}]`, best match first.
    `score` is cosine similarity (higher = closer)."""
    chunks: list[dict] = []
    for m in _get(response, "matches", None) or []:
        meta = dict(_get(m, "metadata", None) or {})
        chunks.append({"text": meta.pop("text", ""), "metadata": meta, "score": _get(m, "score")})
    return chunks


# --- Filing parsing + chunking ---------------------------------------------


def _is_primary(doc_type: str, filing_type: str) -> bool:
    """The form itself (incl. amendments such as 10-K/A) or an EX-99 press release."""
    t = doc_type.upper()
    return (
        t == filing_type or t.startswith(f"{filing_type}/") or t.startswith(PRIMARY_EXHIBIT_PREFIX)
    )


def _primary_documents(raw: str, filing_type: str) -> list[str]:
    """Return the <TEXT> bodies of the primary document(s) in an SGML submission.

    A full-submission.txt bundles the form plus every attachment as
    `<DOCUMENT><TYPE>…<TEXT>…</TEXT></DOCUMENT>` blocks. Only the form (and
    EX-99 press releases) carry narrative worth embedding. Falls back to the
    first block when no type matches, and to the whole file when it has no
    <DOCUMENT> envelope at all.
    """
    docs = _DOCUMENT_RE.findall(raw)
    if not docs:
        return [raw]

    def _body(doc: str) -> str:
        m = _TEXT_RE.search(doc)
        return m.group(1) if m else doc

    kept: list[str] = []
    for doc in docs:
        m = _TYPE_RE.search(doc)
        if m and _is_primary(m.group(1), filing_type):
            kept.append(_body(doc))
    return kept or [_body(docs[0])]


def _extract_text(filing_path: Path) -> str:
    filing_type, _ = _filing_meta_from_path(filing_path)
    raw = filing_path.read_text(errors="ignore")
    parts: list[str] = []
    for doc in _primary_documents(raw, filing_type.upper()):
        parts.append(BeautifulSoup(doc, "html.parser").get_text(separator="\n", strip=True))
    return "\n".join(p for p in parts if p)


def _split_into_items(text: str) -> list[tuple[str, str, str]]:
    """Split filing text on Item headers.

    Returns list of (item_code, item_label, body). If no headers found, returns a
    single ("misc", "Unstructured", text) entry.
    """
    matches = list(ITEM_HEADER_RE.finditer(text))
    if not matches:
        return [("misc", "Unstructured", text)]
    items: list[tuple[str, str, str]] = []
    for i, m in enumerate(matches):
        code = m.group(1).upper()
        label = f"Item {code}. {m.group(2).strip()}".rstrip(". ").rstrip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            items.append((code, label, body))
    return items


def _chunk_tokens(text: str, encoder) -> list[str]:
    tokens = encoder.encode(text)
    if not tokens:
        return []
    step = TARGET_CHUNK_TOKENS - CHUNK_OVERLAP_TOKENS
    chunks: list[str] = []
    for start in range(0, len(tokens), step):
        end = start + TARGET_CHUNK_TOKENS
        chunks.append(encoder.decode(tokens[start:end]))
        if end >= len(tokens):
            break
    return chunks


def _normalize_item_filter(item_filter: str) -> str:
    m = re.match(r"\s*(?:item\s+)?(\d{1,2}[A-Z]?)", item_filter.strip(), re.IGNORECASE)
    return m.group(1).upper() if m else item_filter.strip().upper()


def _filing_meta_from_path(path: Path) -> tuple[str, str]:
    parts = path.parts
    filing_type = parts[-3] if len(parts) >= 3 else "unknown"
    accession = parts[-2] if len(parts) >= 2 else "unknown"
    return filing_type, accession


def _date_int(iso: str) -> int:
    """'2026-02-26' → 20260226. Empty / malformed → 0, which never satisfies
    the `$gte: 1` half of the backtest filter (undated chunks stay out)."""
    digits = (iso or "").replace("-", "")
    return int(digits) if len(digits) == 8 and digits.isdigit() else 0


# --- Filings ingest --------------------------------------------------------


def ingest_filing(ticker: str, filing_path: Path) -> int:
    """Chunk a filing, embed via OpenRouter, upsert into the ticker's namespace.

    Idempotent: a filing whose chunk count already matches the SQLite manifest
    is skipped without an embed call; anything else (never ingested, killed
    mid-run, or chunked differently by a newer parser) has its prior vectors
    dropped by id prefix and is embedded again in full. Returns the number of
    chunks written (0 when skipped).
    """
    ticker = ticker.upper()
    text = _extract_text(filing_path)
    if not text:
        logger.warning(f"{filing_path}: empty extracted text, skipping")
        return 0

    filing_type, accession = _filing_meta_from_path(filing_path)
    filed_date = parse_filed_date(filing_path) or ""  # metadata can't hold None
    encoder = tiktoken.get_encoding(TOKENIZER)

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    chunk_idx = 0  # global per-filing index — keeps IDs unique even when an Item header
    # appears twice (e.g., once in the TOC and once in the body)

    capped = False
    for code, label, body in _split_into_items(text):
        if chunk_idx >= _MAX_CHUNKS_PER_FILING:
            capped = True
            break
        for chunk in _chunk_tokens(body, encoder):
            if chunk_idx >= _MAX_CHUNKS_PER_FILING:
                capped = True
                break
            ids.append(f"{ticker}-{accession}-{chunk_idx}")
            docs.append(chunk)
            metas.append(
                {
                    "ticker": ticker,
                    "filing_type": filing_type,
                    "accession": accession,
                    "filed_date": filed_date,
                    "filed_date_int": _date_int(filed_date),
                    "item_code": code,
                    "item_label": label,
                }
            )
            chunk_idx += 1

    if not docs:
        return 0

    if capped:
        logger.warning(
            f"{ticker} {filing_type} {accession}: hit chunk cap "
            f"({_MAX_CHUNKS_PER_FILING}) — truncating the tail of the filing"
        )

    if state_db.ingested_filing_chunks(ticker, accession) == len(ids):
        logger.info(
            f"{ticker} {filing_type} {accession}: already ingested "
            f"({len(ids)} chunks) — skipping embed"
        )
        return 0

    index = _index(INDEX_FILINGS)
    removed = _delete_by_prefix(index, f"{ticker}-{accession}-", namespace=ticker)
    if removed:
        logger.info(f"{ticker} {accession}: replaced {removed} stale chunks")

    _upsert(index, namespace=ticker, ids=ids, docs=docs, metas=metas)

    state_db.record_ingested_filing(
        ticker=ticker,
        accession=accession,
        filing_type=filing_type,
        filed_date=filed_date,
        chunks=len(ids),
    )
    logger.info(f"Ingested {ticker} {filing_type} {accession}: {len(docs)} chunks")
    return len(docs)


# --- Hybrid retrieval ------------------------------------------------------


@lru_cache(maxsize=4096)
def _tokenise(text: str) -> tuple[str, ...]:
    """Word-tokenise + lowercase + drop stopwords. Cached because the same chunk
    is tokenised once per BM25 call; the question is tokenised many times across
    subqueries during a drill-in."""
    return tuple(t for t in _WORD_RE.findall(text.lower()) if t not in BM25_STOPWORDS)


def _bm25_rank(documents: list[str], question: str) -> list[int]:
    """Return indices into `documents` ranked by BM25 score (descending).

    Uses a regex word-tokeniser plus a small English stopword list. This is
    deliberately lighter than NLTK (no 50MB corpus download, no runtime
    network requirement) but sharper than naive whitespace splitting because
    it strips punctuation and removes high-frequency function words that
    would otherwise dilute the IDF signal.
    """
    if not documents:
        return []
    tokenised_docs = [list(_tokenise(doc)) for doc in documents]
    bm25 = BM25Okapi(tokenised_docs)
    scores = bm25.get_scores(list(_tokenise(question)))
    return sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)


def _reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int = RRF_K) -> list[int]:
    """Merge multiple ranked lists via reciprocal rank fusion.

    For each item, RRF score = Σ 1 / (k + rank_i) over each ranking i.
    `rank_i` is 1-based (best rank = 1). Returns indices sorted by RRF descending.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.keys(), key=lambda i: scores[i], reverse=True)


def _build_filter(item_filter: str | None, as_of: str | None) -> dict | None:
    """Metadata filter applied server-side before the similarity scan.

    The ticker is not part of it — it selects the namespace. Backtest mode
    (`as_of`) keeps only chunks with `1 ≤ filed_date_int ≤ as_of`, so
    undated chunks are excluded (same conservative posture as
    `data/edgar._existing_filings`).
    """
    conds: dict = {}
    if item_filter:
        conds["item_code"] = {"$eq": _normalize_item_filter(item_filter)}
    if as_of:
        conds["filed_date_int"] = {"$gte": 1, "$lte": _date_int(as_of)}
    return conds or None


def query(
    ticker: str | None,
    question: str,
    k: int = 8,
    item_filter: str | None = None,
    *,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    use_keyword: bool = True,
    as_of: str | None = None,
) -> list[dict]:
    """Hybrid retrieval: metadata pre-filter → semantic top-N → BM25 → RRF → top-k.

    Each returned chunk is `{"text": str, "metadata": dict, "score": float|None}`.
    `metadata` includes `filed_date` so downstream agents can reason about
    freshness. `ticker` is required: the filings index is namespaced per ticker.
    """
    if not ticker:
        raise ValueError("query() needs a ticker — the filings index is namespaced per ticker")
    index = _index(INDEX_FILINGS)
    vector = embed_texts([question])[0]
    response = _query_index(
        index,
        namespace=ticker.upper(),
        vector=vector,
        top_k=candidate_pool,
        filter=_build_filter(item_filter, as_of),
    )
    candidates = _unpack_matches(response)
    if not candidates:
        return []

    # Matches arrive best-first, so [0, 1, 2, …] *is* the semantic ranking.
    semantic_ranking = list(range(len(candidates)))
    if use_keyword and len(candidates) > 1:
        keyword_ranking = _bm25_rank([c["text"] for c in candidates], question)
        fused = _reciprocal_rank_fusion([semantic_ranking, keyword_ranking])
    else:
        fused = semantic_ranking
    return [candidates[i] for i in fused[:k]]


# --- Ingest manifest probes (SQLite, no network) ---------------------------


def has_ticker(ticker: str) -> bool:
    """True when at least one filing for `ticker` has been ingested.

    Distinguishes "ticker isn't ingested yet → run scripts/ingest_universe"
    from "the question doesn't match any retrieved chunks". Reads the SQLite
    manifest, so it is safe to call for every ticker on every page load."""
    if not ticker:
        return False
    try:
        return bool(state_db.ingested_filings(ticker.upper()))
    except Exception as e:
        logger.warning(f"[vectors.has_ticker] check failed for {ticker}: {e}")
        return False


def last_filings_by_type(ticker: str) -> dict[str, str]:
    """Return {filing_type: most_recent_filed_date} across `ticker`'s ingested
    filings. Dates are ISO strings parsed from the SEC SGML header at ingest
    time; filings whose date failed to parse are ignored. Empty dict when
    nothing is ingested."""
    if not ticker:
        return {}
    try:
        rows = state_db.ingested_filings(ticker.upper())
    except Exception as e:
        logger.warning(f"[vectors.last_filings_by_type] {ticker}: {e}")
        return {}
    latest: dict[str, str] = {}
    for row in rows:
        ftype, fdate = row.get("filing_type"), row.get("filed_date")
        if ftype and fdate and fdate > latest.get(ftype, ""):
            latest[ftype] = fdate
    return latest


def last_filing_date(ticker: str) -> str | None:
    """Max `filed_date` across all of `ticker`'s ingested filings, or None."""
    by_type = last_filings_by_type(ticker)
    return max(by_type.values()) if by_type else None


# --- Synthesis-report corpus (CIO planner RAG) -----------------------------


def upsert_reports(ids: list[str], docs: list[str], metas: list[dict]) -> int:
    """Embed + upsert report sections into the reports index. Metadata values
    must be primitives; `text` is stored alongside so retrieval can return it."""
    index = _index(INDEX_REPORTS)
    _upsert(index, namespace=REPORTS_NAMESPACE, ids=ids, docs=docs, metas=metas)
    return len(ids)


def query_reports(question: str, *, where: dict | None = None, top_k: int) -> list[dict]:
    """Semantic top-`top_k` report sections, best first, with an optional
    Pinecone metadata filter (`{"ticker": "NVDA"}` or `{"$and": [...]}`)."""
    index = _index(INDEX_REPORTS)
    vector = embed_texts([question])[0]
    return _unpack_matches(
        _query_index(index, namespace=REPORTS_NAMESPACE, vector=vector, top_k=top_k, filter=where)
    )


def fetch_reports(
    ticker: str,
    *,
    thesis: str | None = None,
    section: str | None = None,
    limit: int = 200,
) -> list[dict]:
    """Every indexed section for `ticker` (optionally one thesis / one section),
    in id order — callers sort by date. Lists ids by the `{TICKER}__` prefix
    then fetches metadata, so no embedding call is needed."""
    index = _index(INDEX_REPORTS)
    ids = _list_ids(index, prefix=f"{ticker.upper()}__", namespace=REPORTS_NAMESPACE)
    out: list[dict] = []
    for i in range(0, len(ids), ID_BATCH_SIZE):
        resp = index.fetch(ids=ids[i : i + ID_BATCH_SIZE], namespace=REPORTS_NAMESPACE)
        for v in (_get(resp, "vectors", None) or {}).values():
            meta = dict(_get(v, "metadata", None) or {})
            if thesis and meta.get("thesis") != thesis:
                continue
            if section and meta.get("section") != section:
                continue
            out.append({"text": meta.pop("text", ""), "metadata": meta, "score": None})
            if len(out) >= limit:
                return out
    return out
