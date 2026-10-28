"""Run retrieval and answer evaluations and compare configurations."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Callable, Sequence

from .chunking import Chunk, Chunker
from .data import DataError, Doc, Prediction, Query
from .metrics import Judge, hit_at_k, lexical_groundedness, mrr, ndcg_at_k, precision_at_k, rank_docs, recall_at_k, token_f1, token_recall
from .retrievers import Retriever

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)


@dataclass
class QueryResult:
    id: str
    question: str
    relevant: list[str]
    #: ranked, de-duplicated document ids (at most ``max(ks)``)
    retrieved: list[str]
    metrics: dict[str, float]


@dataclass
class RunResult:
    label: str
    chunker: str
    retriever: str
    ks: tuple[int, ...]
    num_chunks: int
    queries: list[QueryResult]
    summary: dict[str, float]

    @property
    def primary_k(self) -> int:
        return 5 if 5 in self.ks else max(self.ks)

    @property
    def score(self) -> float:
        """Ranking key for comparisons: nDCG at the primary k."""
        return self.summary[f"ndcg@{self.primary_k}"]

    def failures(self) -> list[QueryResult]:
        """Queries with no relevant document in the top ``primary_k``."""
        return [q for q in self.queries if q.metrics[f"hit@{self.primary_k}"] == 0.0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "chunker": self.chunker,
            "retriever": self.retriever,
            "ks": list(self.ks),
            "num_chunks": self.num_chunks,
            "primary_k": self.primary_k,
            "summary": self.summary,
            "queries": [
                {"id": q.id, "question": q.question, "relevant": q.relevant, "retrieved": q.retrieved, "metrics": q.metrics}
                for q in self.queries
            ],
        }


def _validate_ks(ks: Sequence[int]) -> tuple[int, ...]:
    out = tuple(sorted(set(ks)))
    if not out or out[0] < 1:
        raise ValueError("ks must be a non-empty collection of positive integers")
    return out


def evaluate_retrieval(
    corpus: Sequence[Doc],
    dataset: Sequence[Query],
    chunker: Chunker,
    retriever: Retriever,
    ks: Sequence[int] = DEFAULT_KS,
    overfetch: int = 5,
) -> RunResult:
    """Chunk the corpus, index it, run every query and aggregate document-level metrics."""
    ks = _validate_ks(ks)
    kmax = ks[-1]
    chunks: list[Chunk] = []
    for doc in corpus:
        chunks.extend(chunker.split(doc.id, doc.text))
    if not chunks:
        raise DataError("chunking produced no chunks (is the corpus empty?)")
    retriever.index(chunks)

    # Several chunks of one document can occupy the top ranks; fetch extra chunks so
    # that `kmax` distinct documents are usually available after de-duplication.
    fetch = max(kmax * overfetch, 50)
    results: list[QueryResult] = []
    for q in dataset:
        ranked = rank_docs(retriever.search(q.question, fetch))[:kmax]
        m: dict[str, float] = {}
        for k in ks:
            m[f"hit@{k}"] = hit_at_k(ranked, q.relevant, k)
            m[f"recall@{k}"] = recall_at_k(ranked, q.relevant, k)
            m[f"precision@{k}"] = precision_at_k(ranked, q.relevant, k)
            m[f"ndcg@{k}"] = ndcg_at_k(ranked, q.relevant, k)
        m["mrr"] = mrr(ranked, q.relevant, kmax)
        results.append(QueryResult(q.id, q.question, list(q.relevant), ranked, m))

    keys = results[0].metrics.keys()
    summary = {key: fmean(r.metrics[key] for r in results) for key in keys}
    return RunResult(
        label=f"{chunker.name} | {retriever.name}",
        chunker=chunker.name,
        retriever=retriever.name,
        ks=ks,
        num_chunks=len(chunks),
        queries=results,
        summary=summary,
    )


RetrieverOrFactory = Retriever | Callable[[], Retriever]


def _make(r: RetrieverOrFactory) -> Retriever:
    """Accept a ready retriever instance, or a factory (a class or zero-argument callable)."""
    if isinstance(r, type) or not hasattr(r, "search"):
        return r()  # type: ignore[operator, return-value]
    return r  # type: ignore[return-value]


def compare(
    corpus: Sequence[Doc],
    dataset: Sequence[Query],
    chunkers: Sequence[Chunker],
    retrievers: Sequence[RetrieverOrFactory],
    ks: Sequence[int] = DEFAULT_KS,
) -> list[RunResult]:
    """Evaluate every chunker x retriever pair; best first (by nDCG at the primary k, then MRR)."""
    runs = [evaluate_retrieval(corpus, dataset, c, _make(r), ks) for c in chunkers for r in retrievers]
    runs.sort(key=lambda r: (-r.score, -r.summary["mrr"], r.label))
    return runs


# ----------------------------------------------------------------------------- answers


@dataclass
class AnswerResult:
    id: str
    question: str
    answer: str
    token_f1: float | None
    groundedness: float | None
    judge: float | None = None
    token_recall: float | None = None


@dataclass
class GenerationResult:
    queries: list[AnswerResult]
    summary: dict[str, float] = field(default_factory=dict)
    #: dataset queries without a prediction (only non-empty with ``allow_missing``)
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "missing": self.missing,
            "queries": [
                {
                    "id": q.id,
                    "question": q.question,
                    "answer": q.answer,
                    "token_f1": q.token_f1,
                    "token_recall": q.token_recall,
                    "groundedness": q.groundedness,
                    "judge": q.judge,
                }
                for q in self.queries
            ],
        }


def evaluate_answers(
    dataset: Sequence[Query],
    predictions: Sequence[Prediction],
    judge: Judge | None = None,
    allow_missing: bool = False,
) -> GenerationResult:
    """Score answers produced by your RAG system.

    ``token_f1`` and ``token_recall`` need a ``reference_answer`` in the dataset, ``groundedness`` needs ``contexts``
    in the prediction, ``judge`` runs only if you pass one. Metrics that cannot be computed for
    a query are ``None`` and excluded from the averages.
    """
    by_id = {q.id: q for q in dataset}
    unknown = [p.id for p in predictions if p.id not in by_id]
    if unknown:
        raise DataError(f"predictions reference unknown query id(s): {', '.join(unknown[:10])}")
    pred_by_id = {p.id: p for p in predictions}
    missing = [q.id for q in dataset if q.id not in pred_by_id]
    if missing and not allow_missing:
        raise DataError(f"no prediction for {len(missing)} query id(s): {', '.join(missing[:10])} (use allow_missing to skip them)")

    results: list[AnswerResult] = []
    for q in dataset:
        p = pred_by_id.get(q.id)
        if p is None:
            continue
        f1 = token_f1(p.answer, q.reference_answer) if q.reference_answer else None
        recall = token_recall(p.answer, q.reference_answer) if q.reference_answer else None
        grounded = lexical_groundedness(p.answer, p.contexts) if p.contexts else None
        scored = judge.score(q.question, p.answer, p.contexts, q.reference_answer) if judge else None
        results.append(AnswerResult(q.id, q.question, p.answer, f1, grounded, scored, recall))

    summary: dict[str, float] = {"answered": float(len(results)), "coverage": len(results) / len(dataset)}
    for name in ("token_f1", "token_recall", "groundedness", "judge"):
        values = [getattr(r, name) for r in results if getattr(r, name) is not None]
        if values:
            summary[name if name != "judge" else f"judge:{judge.name}"] = fmean(values)  # type: ignore[union-attr]
    return GenerationResult(results, summary, missing)
