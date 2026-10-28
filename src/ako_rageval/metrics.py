"""Metrics.

Retrieval metrics are computed at *document* level: ranked chunks are mapped to their
parent documents, duplicates are dropped (first occurrence wins) and the top ``k``
documents are compared with the relevant set. This makes different chunkers comparable.

Answer metrics here are **lexical baselines** (token overlap). They are cheap, deterministic
and useful for regression tracking, but they are not a substitute for human review or an
LLM judge; plug one in through :class:`Judge`.
"""

from __future__ import annotations

import math
from typing import Iterable, Protocol, Sequence

from .retrievers import Hit
from .tokenizer import content_tokens


def rank_docs(hits: Iterable[Hit]) -> list[str]:
    """Document ids in rank order, each document once."""
    seen: set[str] = set()
    ranked: list[str] = []
    for hit in hits:
        doc = hit.chunk.doc_id
        if doc not in seen:
            seen.add(doc)
            ranked.append(doc)
    return ranked


def _top(ranked: Sequence[str], k: int) -> Sequence[str]:
    if k < 1:
        raise ValueError("k must be >= 1")
    return ranked[:k]


def hit_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    return 1.0 if any(d in rel for d in _top(ranked, k)) else 0.0


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        raise ValueError("relevant must not be empty")
    return len(rel.intersection(_top(ranked, k))) / len(rel)


def precision_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    return sum(1 for d in _top(ranked, k) if d in rel) / k


def mrr(ranked: Sequence[str], relevant: Iterable[str], k: int | None = None) -> float:
    rel = set(relevant)
    for i, d in enumerate(ranked if k is None else _top(ranked, k), start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """nDCG with binary relevance."""
    rel = set(relevant)
    if not rel:
        raise ValueError("relevant must not be empty")
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(_top(ranked, k)) if d in rel)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / ideal


# ----------------------------------------------------------------------------- answers


def _overlap(prediction: str, reference: str) -> tuple[int, int, int]:
    """(common, len(pred), len(ref)) over normalized content tokens, counting multiplicity."""
    pred, ref = content_tokens(prediction), content_tokens(reference)
    remaining: dict[str, int] = {}
    for t in ref:
        remaining[t] = remaining.get(t, 0) + 1
    common = 0
    for t in pred:
        if remaining.get(t, 0) > 0:
            remaining[t] -= 1
            common += 1
    return common, len(pred), len(ref)


def token_f1(prediction: str, reference: str) -> float:
    """F1 over normalized content tokens (multiset overlap), SQuAD-style.

    Penalizes verbose answers: "تخت جمشید در نزدیکی شیراز است" scores low against the
    reference "شیراز". Prefer :func:`token_recall` when references are short.
    """
    common, n_pred, n_ref = _overlap(prediction, reference)
    if common == 0:
        return 0.0
    precision, recall = common / n_pred, common / n_ref
    return 2 * precision * recall / (precision + recall)


def token_recall(prediction: str, reference: str) -> float:
    """Share of the reference's content tokens that appear in the answer (verbosity is not penalized)."""
    common, _, n_ref = _overlap(prediction, reference)
    return common / n_ref if n_ref else 0.0


def lexical_groundedness(answer: str, contexts: Iterable[str]) -> float:
    """Fraction of the answer's content tokens that appear in the retrieved contexts.

    A low value flags answers that use words absent from the evidence (possible
    hallucination); a high value does not prove the answer is correct or entailed.
    """
    tokens = content_tokens(answer)
    if not tokens:
        return 0.0
    support: set[str] = set()
    for c in contexts:
        support.update(content_tokens(c))
    return sum(1 for t in tokens if t in support) / len(tokens)


class Judge(Protocol):
    """Plug an LLM judge (or any scorer) into answer evaluation. Return a score in [0, 1]."""

    name: str

    def score(self, question: str, answer: str, contexts: Sequence[str], reference: str | None) -> float: ...
