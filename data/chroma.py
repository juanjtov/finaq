"""ChromaDB ingest + query for the SEC filings RAG corpus.

Single persistent collection `filings` keyed by chunk id. Metadata holds
{ticker, filing_type, accession, item_code, item_label}. Embeddings come from
OpenRouter (`MODEL_EMBEDDINGS`, default `text-embedding-3-small`).
"""

from __future__ import annotations

import re
from pathlib import Path

import chromadb
import tiktoken
from bs4 import BeautifulSoup
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings

from utils import logger
from utils.models import MODEL_EMBEDDINGS
from utils.openrouter import get_client

CHROMA_DIR = Path("data_cache/chroma")
COLLECTION_NAME = "filings"
TARGET_CHUNK_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 100
EMBED_BATCH_SIZE = 100  # OpenAI embeddings API batch limit
TOKENIZER = "cl100k_base"

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


def _get_collection():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=OpenRouterEmbedding(),
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


def ingest_filing(ticker: str, filing_path: Path) -> int:
    """Chunk a filing, embed via OpenRouter, upsert into the `filings` collection.

    Idempotent — re-running on the same file replaces existing chunks for that
    (ticker, accession) tuple. Returns the number of chunks written.
    """
    text = _extract_text(filing_path)
    if not text:
        logger.warning(f"{filing_path}: empty extracted text, skipping")
        return 0

    filing_type, accession = _filing_meta_from_path(filing_path)
    encoder = tiktoken.get_encoding(TOKENIZER)
    coll = _get_collection()

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    chunk_idx = 0  # global per-filing index — keeps IDs unique even when an Item header
    # appears twice (e.g., once in the TOC and once in the body)

    for code, label, body in _split_into_items(text):
        for chunk in _chunk_tokens(body, encoder):
            ids.append(f"{ticker}-{accession}-{chunk_idx}")
            docs.append(chunk)
            metas.append(
                {
                    "ticker": ticker,
                    "filing_type": filing_type,
                    "accession": accession,
                    "item_code": code,
                    "item_label": label,
                }
            )
            chunk_idx += 1

    if not docs:
        return 0

    coll.upsert(ids=ids, documents=docs, metadatas=metas)
    logger.info(f"Ingested {ticker} {filing_type} {accession}: {len(docs)} chunks")
    return len(docs)


def query(
    ticker: str | None,
    question: str,
    k: int = 8,
    item_filter: str | None = None,
) -> list[dict]:
    """Top-k chunks for a question. Filters by ticker (or None = cross-ticker)
    and optionally an item_filter ("1A", "Item 7", "7A", ...).
    """
    coll = _get_collection()

    conditions: list[dict] = []
    if ticker:
        conditions.append({"ticker": ticker})
    if item_filter:
        conditions.append({"item_code": _normalize_item_filter(item_filter)})

    where = None
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    results = coll.query(query_texts=[question], n_results=k, where=where)

    chunks: list[dict] = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = (results.get("distances") or [[None] * len(ids)])[0]
    for i in range(len(ids)):
        chunks.append({"text": docs[i], "metadata": metas[i], "score": dists[i]})
    return chunks
