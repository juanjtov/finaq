"""Tier 3 RAG evaluation — RAGAS-framework metrics.

RAGAS computes:
  - faithfulness          (every claim in the answer grounded in the contexts)
  - answer_relevancy      (does the answer address the question?)
  - context_precision     (are top contexts truly relevant?)
  - context_recall        (does context cover the ground-truth answer?)

The framework expects a LangChain BaseChatModel + Embeddings. We point those at
OpenRouter using `MODEL_JUDGE` (cheap-tier role) — same model used by Tier 2.

Heavy dependency (langchain + datasets pulled in via ragas). Worth it for
standardised RAG metrics that match published benchmarks.

Usage:

    from utils.rag_ragas import evaluate_filings_run

    metrics = evaluate_filings_run(
        question="...",
        retrieved_chunks=[...],
        answer="...",
        ground_truth="...",
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from utils.models import MODEL_EMBEDDINGS, MODEL_JUDGE

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class RagasReport:
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    context_recall: float | None
    raw: dict  # full RAGAS output for debugging

    def to_dict(self) -> dict:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
        }


def _make_langchain_judge():
    """LangChain ChatOpenAI pointed at OpenRouter, using the configured judge model."""
    # Imported lazily so the rest of utils/ doesn't pay the langchain import cost.
    from langchain_openai import ChatOpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return ChatOpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        model=MODEL_JUDGE,
        temperature=0,
        max_retries=3,
    )


def _make_langchain_embeddings():
    """LangChain OpenAIEmbeddings pointed at OpenRouter."""
    from langchain_openai import OpenAIEmbeddings

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return OpenAIEmbeddings(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        model=MODEL_EMBEDDINGS,
        check_embedding_ctx_length=False,
    )


def evaluate_filings_run(
    question: str,
    retrieved_chunks: list[dict],
    answer: str,
    ground_truth: str | None = None,
) -> RagasReport:
    """Run RAGAS metrics on a single (question, contexts, answer, ground_truth) tuple.

    Returns a RagasReport with the four metric scores. Any metric that requires
    a `ground_truth` (context_recall) will be `None` if `ground_truth` is omitted.
    """
    # Imports kept inside the function so Tier 1/2 don't pay the cost.
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    judge = LangchainLLMWrapper(_make_langchain_judge())
    embeddings = LangchainEmbeddingsWrapper(_make_langchain_embeddings())

    contexts = [c.get("text", "") for c in retrieved_chunks]
    sample = {
        "user_input": question,
        "retrieved_contexts": contexts,
        "response": answer,
    }
    if ground_truth:
        sample["reference"] = ground_truth

    dataset = EvaluationDataset.from_list([sample])

    metrics: list = [
        Faithfulness(llm=judge),
        AnswerRelevancy(llm=judge, embeddings=embeddings),
        ContextPrecision(llm=judge),
    ]
    if ground_truth:
        metrics.append(ContextRecall(llm=judge))

    result = evaluate(dataset=dataset, metrics=metrics)
    raw = result.to_pandas().iloc[0].to_dict()

    def _get(name: str) -> float | None:
        v = raw.get(name)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    return RagasReport(
        faithfulness=_get("faithfulness"),
        answer_relevancy=_get("answer_relevancy"),
        context_precision=_get("context_precision") or _get("llm_context_precision_with_reference"),
        context_recall=_get("context_recall"),
        raw={k: str(v) for k, v in raw.items()},  # stringified for JSON-serialisability
    )
