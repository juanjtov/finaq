"""ChromaDB ingest + hybrid query for the SEC filings RAG corpus.

Single persistent collection `filings` keyed by chunk id. Metadata holds
{ticker, filing_type, accession, filed_date, item_code, item_label}. Distance
metric is **cosine** (explicitly configured — ChromaDB's default is L2). For
unit-normalised text-embedding vectors L2 and cosine produce identical rankings,
but cosine is the semantically correct choice and is robust if we swap to a
non-normalised embedding model later.

Embeddings come from OpenRouter via the env-configured embedding model.

Retrieval is hybrid:
  1. Metadata pre-filter via ChromaDB `where` clause (applied BEFORE similarity).
  2. Semantic search returns top `candidate_pool` chunks (cosine-ranked).
  3. BM25 keyword search runs over the same pool.
  4. Reciprocal Rank Fusion merges both rankings into the final top-k.

Cross-encoder re-ranking is intentionally NOT included; see docs/POSTPONED.md §2.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import chromadb
import tiktoken
from bs4 import BeautifulSoup
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from data.edgar import parse_filed_date
from utils import logger
from utils.models import MODEL_EMBEDDINGS
from utils.openrouter import get_client

CHROMA_DIR = Path("data_cache/chroma")
COLLECTION_NAME = "filings"
"""Default collection name. The chunks-of-SEC-filings RAG corpus.

Other corpora (e.g. `synthesis_reports` for the CIO planner's RAG over
past drill-in reports) live in their own collections in the same on-disk
ChromaDB instance — `_get_collection(name="synthesis_reports")` opens or
creates them. The embedding function + cosine distance config are shared
across collections; only `where`-clause schema differs."""

DISTANCE_SPACE = "cosine"  # explicit — ChromaDB's default is l2
TARGET_CHUNK_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 100
EMBED_BATCH_SIZE = 100  # OpenAI embeddings API batch limit
TOKENIZER = "cl100k_base"
DEFAULT_CANDIDATE_POOL = 60  # semantic top-N, pre-fusion
RRF_K = 60  # standard reciprocal-rank-fusion constant

# Lightweight English stopword list for BM25 tokenisation. Removing high-frequency
# function words sharpens the IDF signal so rare, discriminative terms (e.g.,
# "Blackwell", "capex", "constraint") drive ranking instead of "the"/"of"/"and".
# Curated subset of the canonical NLTK English stopword list (~150 entries).
_BM25_STOPWORDS = frozenset(
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


class OpenRouterEmbedding(EmbeddingFunction[Documents]):
    def __init__(self, model: str = MODEL_EMBEDDINGS) -> None:
        self._model = model
        self._client = get_client()

    def __call__(self, input: Documents) -> Embeddings:
        out: list[list[float]] = []
        for i in range(0, len(input), EMBED_BATCH_SIZE):
            batch = list(input[i : i + EMBED_BATCH_SIZE])
            resp = self._client.embeddings.create(model=self._model, input=batch)
            out.extend([d.embedding for d in resp.data])
        return out

    @staticmethod
    def name() -> str:
        return "openrouter"

    def get_config(self) -> dict:
        return {"model": self._model}

    @classmethod
    def build_from_config(cls, config: dict) -> OpenRouterEmbedding:
        return cls(model=config.get("model", MODEL_EMBEDDINGS))


def _get_collection(name: str = COLLECTION_NAME):
    """Open (or create) a persistent ChromaDB collection with cosine distance.

    `name` defaults to `COLLECTION_NAME` ("filings") so all existing call
    sites are unchanged. The CIO planner (Step 11.8) opens
    `name="synthesis_reports"` to RAG over past drill-in reports.

    Note: ChromaDB applies the `configuration` parameter only on collection
    creation. If the on-disk collection was created with a different space
    (e.g., the legacy l2 default), this function returns it unchanged — wipe
    `data_cache/chroma/<collection>` and re-ingest to pick up the cosine config.
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=name,
        embedding_function=OpenRouterEmbedding(),
        configuration={"hnsw": {"space": DISTANCE_SPACE}},
    )


def _extract_text(filing_path: Path) -> str:
    raw = filing_path.read_text(errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")
    return soup.get_text(separator="\n", strip=True)


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


# ChromaDB's `add`/`upsert` API caps batch size around 5461 (it's the
# default max_batch_size from chromadb.config). Filings like SMCI's 10-K
# can produce 7000+ chunks in a single call, which raises ValueError.
# We chunk the upsert into batches well below the cap.
_UPSERT_BATCH_SIZE = 4000

# Hard cap on chunks per filing. NU's first 20-F produced 19,423 chunks
# (35MB of XBRL-inflated text), which blew through 84GB of disk space on
# the HNSW index before being killed mid-ingest. The cap protects against
# pathological filings — a typical 10-K runs 1,500-3,000 chunks; SMCI's
# big-cap 10-K runs ~7,200; so 6,000 leaves the body intact for normal
# filings while preventing runaway 20-Fs (19k+ chunks) from eating the
# disk. When the cap fires we keep the FIRST N chunks (the body of the
# filing — exhibits and XBRL-noise sections come later in the SGML).
_MAX_CHUNKS_PER_FILING = 6000


def ingest_filing(ticker: str, filing_path: Path) -> int:
    """Chunk a filing, embed via OpenRouter, upsert into the `filings` collection.

    Idempotent — re-running on the same file replaces existing chunks for that
    (ticker, accession) tuple. Returns the number of chunks written.

    Large filings (typically big-cap 10-Ks like SMCI / DELL) can produce more
    chunks than ChromaDB's per-call max_batch_size (~5461). We batch the
    upsert in `_UPSERT_BATCH_SIZE`-sized chunks to stay well below the cap.

    Chunk count is also capped at `_MAX_CHUNKS_PER_FILING` per filing to
    protect against XBRL-inflated 20-F filings (NU's first 20-F generated
    19,423 chunks and ate 84GB of disk before being killed). If the cap
    fires we log a warning and keep the head of the filing (where the
    Items 1-7 prose sits; exhibits and XBRL sections come at the tail).
    """
    text = _extract_text(filing_path)
    if not text:
        logger.warning(f"{filing_path}: empty extracted text, skipping")
        return 0

    filing_type, accession = _filing_meta_from_path(filing_path)
    filed_date = parse_filed_date(filing_path) or ""  # ChromaDB metadata can't hold None
    encoder = tiktoken.get_encoding(TOKENIZER)
    coll = _get_collection()

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
            f"({_MAX_CHUNKS_PER_FILING}); body sections beyond the cap "
            f"(typically exhibits / XBRL) are not indexed. This is the "
            f"defensive behaviour after the NU 20-F disk-fill incident."
        )

    # Batched upsert — stays under ChromaDB's max_batch_size limit.
    for i in range(0, len(docs), _UPSERT_BATCH_SIZE):
        coll.upsert(
            ids=ids[i : i + _UPSERT_BATCH_SIZE],
            documents=docs[i : i + _UPSERT_BATCH_SIZE],
            metadatas=metas[i : i + _UPSERT_BATCH_SIZE],
        )
    logger.info(f"Ingested {ticker} {filing_type} {accession}: {len(docs)} chunks")
    return len(docs)


@lru_cache(maxsize=4096)
def _tokenise(text: str) -> tuple[str, ...]:
    """Word-tokenise + lowercase + drop stopwords. Cached because the same chunk
    is tokenised once per BM25 call; the question is tokenised many times across
    subqueries during a drill-in."""
    return tuple(t for t in _WORD_RE.findall(text.lower()) if t not in _BM25_STOPWORDS)


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


def _build_where_clause(
    ticker: str | None,
    item_filter: str | None,
) -> dict | None:
    """Compose the metadata pre-filter that ChromaDB applies BEFORE similarity scan.

    Note: backtest's `as_of` filter is NOT applied here. ChromaDB's `$lte`
    operator rejects string operands with "Expected operand value to be an
    int or a float for operator $lte" (verified 2026-05-08 against the
    chromadb version pinned in requirements.txt). The historical-cutoff
    filter is applied in Python via `_filter_by_as_of` after retrieval.
    The cost is retrieving slightly more candidates than strictly needed
    (post-as_of chunks land in the candidate pool then get dropped) — at
    DEFAULT_CANDIDATE_POOL = 60 this is negligible, especially since
    backtest is a single-user offline workflow.
    """
    conditions: list[dict] = []
    if ticker:
        conditions.append({"ticker": ticker})
    if item_filter:
        conditions.append({"item_code": _normalize_item_filter(item_filter)})
    if len(conditions) == 0:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _filter_by_as_of(chunks: list[dict], as_of: str | None) -> list[dict]:
    """Backtest mode post-filter: drop chunks whose `filed_date > as_of`.

    Operates on the in-memory chunks returned by ChromaDB instead of the
    `where` clause because ChromaDB's `$lte` requires numeric operands but
    `filed_date` is stored as ISO strings. ISO dates are lexicographically
    ordered, so a string comparison preserves calendar order. Chunks with
    missing or unparseable `filed_date` are DROPPED in backtest mode (same
    conservative posture as `data/edgar._existing_filings`)."""
    if not as_of:
        return chunks
    kept: list[dict] = []
    for c in chunks:
        meta = c.get("metadata") or {}
        filed = (meta.get("filed_date") or "").strip()
        if not filed:
            continue  # conservative: drop undated chunks in backtest
        if filed <= as_of:
            kept.append(c)
    return kept


def _unpack_chroma_query(results: dict) -> list[dict]:
    chunks: list[dict] = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = (results.get("distances") or [[None] * len(ids)])[0]
    for i in range(len(ids)):
        chunks.append({"text": docs[i], "metadata": metas[i], "score": dists[i]})
    return chunks


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
    """Hybrid retrieval: semantic + BM25 + RRF, with metadata pre-filter.

    1. ChromaDB `where` clause (ticker + item_code [+ filed_date ≤ as_of]) is
       applied BEFORE similarity.
    2. Semantic search returns top `candidate_pool` chunks.
    3. BM25 ranks the same pool by keyword score (skipped if `use_keyword=False`).
    4. RRF merges both rankings; top `k` returned.

    Each returned chunk is `{"text": str, "metadata": dict, "score": float|None}`.
    `metadata` includes `filed_date` so downstream agents can reason about freshness.

    Backtest mode (`as_of="YYYY-MM-DD"`): only chunks from filings dated ≤ as_of
    are eligible. The cutoff is enforced via `_filter_by_as_of` after the
    candidate pool is retrieved (ChromaDB's `$lte` requires numeric operands;
    our `filed_date` is an ISO string). No forward leakage — post-as_of
    chunks are dropped before BM25 / RRF run.
    """
    coll = _get_collection()
    where = _build_where_clause(ticker, item_filter)

    # Step 1+2 — pre-filter + semantic candidate pool.
    # ChromaDB internally: (a) embeds the query via our OpenRouterEmbedding,
    # (b) applies the `where` metadata filter, (c) computes cosine distance
    # against every candidate's stored embedding, (d) returns the top-N sorted
    # by distance ascending (closest first). The `score` field on each chunk
    # is the cosine distance — lower = more similar.
    # Backtest mode: pull a wider candidate pool so the post-filter has
    # enough headroom — many of the top-N pre-as_of-filter chunks may be
    # newer than as_of and get dropped. 3x is conservative; doesn't matter
    # for production (as_of=None bypasses _filter_by_as_of entirely).
    n_results = candidate_pool * 3 if as_of else candidate_pool
    sem_results = coll.query(query_texts=[question], n_results=n_results, where=where)
    candidates = _unpack_chroma_query(sem_results)
    if as_of:
        candidates = _filter_by_as_of(candidates, as_of)[:candidate_pool]
    if not candidates:
        return []

    # ChromaDB returns candidates already ordered best-first; a list of indices
    # [0, 1, 2, ...] therefore *is* the semantic ranking (each int is the
    # candidate's position in the array).
    semantic_ranking_indices = list(range(len(candidates)))

    # Step 3 — BM25 over the same pre-filtered pool. RRF then merges the two
    # rankings; if keyword search is disabled we fall through to pure semantic.
    if use_keyword and len(candidates) > 1:
        keyword_ranking_indices = _bm25_rank([c["text"] for c in candidates], question)
        # Step 4 — Reciprocal Rank Fusion merges both rankings.
        fused = _reciprocal_rank_fusion([semantic_ranking_indices, keyword_ranking_indices])
    else:
        fused = semantic_ranking_indices

    return [candidates[i] for i in fused[:k]]


def has_ticker(ticker: str) -> bool:
    """Return True if at least one chunk for `ticker` exists in ChromaDB.

    Distinguishes "ticker isn't ingested yet → run scripts/ingest_universe"
    from "the question doesn't match any retrieved chunks." The Filings
    agent uses this to emit a precise actionable error message instead of
    a generic "no chunks retrieved" warning.

    Implementation: a single-result `get(where={"ticker": ...})` round-trip.
    Cheap (~1ms) — uses the metadata index, no embedding call.
    """
    if not ticker:
        return False
    try:
        coll = _get_collection()
        # `limit=1` short-circuits as soon as ChromaDB finds one matching chunk.
        result = coll.get(where={"ticker": ticker.upper()}, limit=1)
        ids = result.get("ids") or []
        return len(ids) > 0
    except Exception as e:
        # Be conservative: if the check fails (e.g. collection missing),
        # return False so the caller surfaces the "ingest first" hint.
        from utils import logger

        logger.warning(f"[chroma.has_ticker] check failed for {ticker}: {e}")
        return False
