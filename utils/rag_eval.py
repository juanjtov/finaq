"""RAG evaluation utilities — Tier 1, 2, and 3 metrics.

Three tiers, each used by a different test gate:

  Tier 1 (always-on, deterministic)
    - faithfulness via substring match (mdna_quotes appear in retrieved chunks)
    - citation accuracy (every evidence.accession matches a retrieved chunk)
    - golden-set recall@K (see tests/eval/golden_queries.py)

  Tier 2 (`pytest -m eval`, costs ~$0.12/run)
    - LLM-as-judge: per-chunk relevance scoring → precision@K, NDCG@K, MRR

  Tier 3 (`pytest -m eval`, costs ~$1+/run)
    - RAGAS framework: faithfulness, context_precision, context_recall,
      answer_relevancy. See utils/rag_ragas.py.

The judge model is configured via `MODEL_JUDGE` env var (cheap-tier role).

Eval results persist to data_cache/eval/runs/{timestamp}.json so the
Mission Control panel (Step 8) can render them.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from utils import logger
from utils.models import MODEL_JUDGE
from utils.openrouter import get_client

EVAL_OUTPUT_DIR = Path("data_cache/eval/runs")
JUDGE_MAX_TOKENS = 200


# --- Tier 1: deterministic, always-on ---------------------------------------


@dataclass(frozen=True)
class FaithfulnessResult:
    quotes_total: int
    quotes_grounded: int  # found verbatim in retrieved chunks
    citations_total: int
    citations_grounded: int  # accession matches a retrieved chunk
    faithfulness_rate: float  # quotes_grounded / max(quotes_total, 1)
    citation_accuracy: float
    ungrounded_quotes: tuple[str, ...]
    fabricated_accessions: tuple[str, ...]


def _normalise_text(s: str) -> str:
    """Collapse whitespace + lowercase for fuzzy substring comparison.
    Filings often have ragged whitespace from BeautifulSoup; verbatim quotes
    from the LLM may use single newlines while chunks have doubles."""
    return re.sub(r"\s+", " ", s).strip().lower()


def _alnum_only(s: str) -> str:
    """Strip everything except letters and digits, then lowercase. Used for
    fuzzy faithfulness checking — tolerates the LLM normalising punctuation,
    spaces, smart-quotes, etc. while still catching fabricated quotes (which
    have no alphanumeric overlap with any retrieved chunk)."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


# A quote is considered grounded if at least the first FAITHFULNESS_PREFIX_CHARS
# alphanumeric characters of the quote appear inside the alphanumeric-only form
# of any retrieved chunk. Catches real fabrication (no overlap at all) while
# tolerating LLM-style light paraphrasing in the tail of long quotes.
FAITHFULNESS_PREFIX_CHARS = 60


def check_faithfulness(
    mdna_quotes: list[dict],
    evidence: list[dict],
    retrieved_chunks: list[dict],
) -> FaithfulnessResult:
    """Tier-1 deterministic faithfulness + citation-accuracy check.

    A quote is `grounded` if the first 60 alphanumeric characters of its
    normalised text appear inside any retrieved chunk's alphanumeric-only
    text. A citation is `grounded` if its `accession` matches a retrieved
    chunk's `accession`.
    """
    chunk_texts_alnum = [_alnum_only(c.get("text", "")) for c in retrieved_chunks]
    chunk_accessions = {c.get("metadata", {}).get("accession", "") for c in retrieved_chunks}

    ungrounded_quotes: list[str] = []
    quotes_grounded = 0
    for q in mdna_quotes or []:
        text = q.get("text", "") or ""
        needle_alnum = _alnum_only(text)[:FAITHFULNESS_PREFIX_CHARS]
        if not needle_alnum:
            continue
        if any(needle_alnum in c for c in chunk_texts_alnum):
            quotes_grounded += 1
        else:
            ungrounded_quotes.append(text[:80])

    fabricated_accessions: list[str] = []
    citations_grounded = 0
    for ev in evidence or []:
        acc = ev.get("accession") or ""
        if not acc:
            continue
        if acc in chunk_accessions:
            citations_grounded += 1
        else:
            fabricated_accessions.append(acc)

    quotes_total = len([q for q in (mdna_quotes or []) if q.get("text")])
    citations_total = len([e for e in (evidence or []) if e.get("accession")])

    # Vacuous truth: no claims to check → 100% faithful (nothing was hallucinated).
    faithfulness_rate = (quotes_grounded / quotes_total) if quotes_total else 1.0
    citation_accuracy = (citations_grounded / citations_total) if citations_total else 1.0

    return FaithfulnessResult(
        quotes_total=quotes_total,
        quotes_grounded=quotes_grounded,
        citations_total=citations_total,
        citations_grounded=citations_grounded,
        faithfulness_rate=faithfulness_rate,
        citation_accuracy=citation_accuracy,
        ungrounded_quotes=tuple(ungrounded_quotes),
        fabricated_accessions=tuple(fabricated_accessions),
    )


# --- Tier 2: LLM-as-judge ---------------------------------------------------

# Categorical labels are more robust than integer scales — the judge model
# recognises semantic categories from its training data rather than inferring
# blurry numeric boundaries (a 7B/70B model's "2" might be a frontier model's
# "3"). We map labels to integer scores internally for graded NDCG@K math.
_LABEL_TO_SCORE: dict[str, int] = {
    "NONE": 0,
    "WEAK": 1,
    "PARTIAL": 2,
    "HIGH": 3,
}
_RELEVANT_THRESHOLD = 2  # >= this counts as relevant for precision@K and MRR

# IMPORTANT prompt-engineering details below:
#  1. `rationale` appears BEFORE `label` in the JSON. LLMs are autoregressive,
#     so requiring the rationale first forces a mini chain-of-thought that
#     improves the final label quality. Putting label first would be a blind
#     guess that the model then justifies post-hoc.
#  2. Categorical labels (NONE/WEAK/PARTIAL/HIGH) are more model-portable than
#     integer scores. Smaller / cheaper / open-source judge models handle these
#     reliably; integer 0-3 boundaries drift across models.
_JUDGE_PROMPT = """You are a relevance grader for a financial-research RAG system.

You will be given a USER QUESTION and a single CANDIDATE CHUNK retrieved from
SEC filings. Classify the chunk's relevance to the question using exactly one
of these labels:

  NONE     Off-topic. Does not address the question.
  WEAK     Tangentially related. Mentions topic surface-area but no useful content.
  PARTIAL  Directly relates but is incomplete or weak.
  HIGH     Substantive content that directly addresses the question.

Output STRICT JSON, no prose, no markdown fences. Use exactly this schema, in
this key order so you reason BEFORE committing to a label:

{"rationale": "<one short sentence explaining your judgment>", "label": "<NONE|WEAK|PARTIAL|HIGH>"}
"""


@dataclass
class JudgeScore:
    chunk_index: int
    label: str  # NONE | WEAK | PARTIAL | HIGH
    score: int  # 0-3, mapped from label via _LABEL_TO_SCORE
    rationale: str


@dataclass
class JudgeReport:
    question: str
    k: int
    scores: list[JudgeScore]
    precision_at_k: float  # fraction with score >= _RELEVANT_THRESHOLD (PARTIAL or HIGH)
    ndcg_at_k: float
    mrr: float  # 1/rank of first relevant chunk, else 0


def _label_to_score(raw_label: str) -> int:
    """Map a raw judge label to an integer score. Defaults to 0 (NONE) for
    unrecognised labels — a noisy judge response counts as not-relevant."""
    return _LABEL_TO_SCORE.get((raw_label or "").strip().upper(), 0)


def _strip_judge_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        nl = raw.find("\n")
        if nl > 0:
            raw = raw[nl + 1 :]
        if raw.endswith("```"):
            raw = raw[:-3].rstrip()
    return raw


def _judge_one_chunk(question: str, chunk_text: str) -> tuple[str, int, str]:
    """Single LLM call to grade one chunk. Returns (label, score, rationale)."""
    client = get_client()
    user = f"USER QUESTION:\n{question}\n\nCANDIDATE CHUNK:\n{chunk_text[:2000]}"
    resp = client.chat.completions.create(
        model=MODEL_JUDGE,
        messages=[
            {"role": "system", "content": _JUDGE_PROMPT},
            {"role": "user", "content": user},
        ],
        max_tokens=JUDGE_MAX_TOKENS,
    )
    raw = _strip_judge_fences(resp.choices[0].message.content or "")
    try:
        data = json.loads(raw)
        label_raw = str(data.get("label", "")).strip().upper()
        # Defensive: older deployments may still emit `score` instead of `label`.
        # Translate integer back to label so the rest of the pipeline is uniform.
        if not label_raw and "score" in data:
            score_int = max(0, min(3, int(data.get("score", 0))))
            label_raw = next((lbl for lbl, s in _LABEL_TO_SCORE.items() if s == score_int), "NONE")
        if label_raw not in _LABEL_TO_SCORE:
            label_raw = "NONE"
        rationale = str(data.get("rationale", ""))
        return label_raw, _label_to_score(label_raw), rationale
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning(f"[judge] failed to parse: {raw[:120]} ({e})")
        return "NONE", 0, "judge response unparseable"


def judge_relevance(question: str, chunks: list[dict]) -> JudgeReport:
    """Score every chunk for relevance to `question`. Computes precision@K,
    NDCG@K, and MRR. One LLM call per chunk."""
    scores: list[JudgeScore] = []
    for i, chunk in enumerate(chunks):
        label, score, rationale = _judge_one_chunk(question, chunk.get("text", ""))
        scores.append(JudgeScore(chunk_index=i, label=label, score=score, rationale=rationale))

    relevant_count = sum(1 for s in scores if s.score >= _RELEVANT_THRESHOLD)
    precision = relevant_count / max(len(chunks), 1)

    # NDCG@K with graded relevance (DCG / IDCG)
    def dcg(rel_seq: list[int]) -> float:
        return sum(r / math.log2(i + 2) for i, r in enumerate(rel_seq))

    actual_rels = [s.score for s in scores]
    ideal_rels = sorted(actual_rels, reverse=True)
    ndcg = dcg(actual_rels) / dcg(ideal_rels) if dcg(ideal_rels) > 0 else 0.0

    # MRR: 1/rank of first chunk at PARTIAL or HIGH
    mrr = 0.0
    for i, s in enumerate(scores, start=1):
        if s.score >= _RELEVANT_THRESHOLD:
            mrr = 1.0 / i
            break

    return JudgeReport(
        question=question,
        k=len(chunks),
        scores=scores,
        precision_at_k=precision,
        ndcg_at_k=ndcg,
        mrr=mrr,
    )


# --- Persistence: Mission Control reads from these JSON files --------------


def write_eval_run(run: dict[str, Any]) -> Path:
    """Persist an eval run's results so the Mission Control panel (Step 8) can
    render them. Returns the path written to."""
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    run_with_meta = {
        "timestamp": datetime.now(UTC).isoformat(),
        "epoch": time.time(),
        **run,
    }
    path = EVAL_OUTPUT_DIR / f"{ts}.json"
    path.write_text(json.dumps(run_with_meta, indent=2, default=str))
    return path


def serialise_judge_report(report: JudgeReport) -> dict:
    """Convert JudgeReport into a JSON-friendly dict for write_eval_run()."""
    return {
        "question": report.question,
        "k": report.k,
        "precision_at_k": report.precision_at_k,
        "ndcg_at_k": report.ndcg_at_k,
        "mrr": report.mrr,
        "scores": [asdict(s) for s in report.scores],
    }
