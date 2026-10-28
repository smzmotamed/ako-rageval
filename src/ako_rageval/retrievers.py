"""Retrievers: a BM25 baseline, a pluggable embedding retriever and rank fusion."""

from __future__ import annotations

import importlib
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from .chunking import Chunk
from .tokenizer import Tokenizer


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    name: str

    def index(self, chunks: Sequence[Chunk]) -> None: ...

    def search(self, query: str, k: int) -> list[Hit]: ...


class BM25Retriever:
    """Okapi BM25 over chunks (pure Python, no dependencies)."""

    def __init__(self, tokenizer: Tokenizer | None = None, k1: float = 1.5, b: float = 0.75, name: str | None = None):
        self.tokenizer = tokenizer or Tokenizer()
        self.k1 = k1
        self.b = b
        self.name = name or f"bm25-{self.tokenizer.name}"
        self._chunks: list[Chunk] = []
        self._tf: list[Counter[str]] = []
        self._len: list[int] = []
        self._postings: dict[str, list[int]] = defaultdict(list)
        self._idf: dict[str, float] = {}
        self._avgdl = 0.0

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        self._tf = []
        self._len = []
        self._postings = defaultdict(list)
        for i, ch in enumerate(self._chunks):
            toks = self.tokenizer(ch.text)
            tf = Counter(toks)
            self._tf.append(tf)
            self._len.append(len(toks))
            for term in tf:
                self._postings[term].append(i)
        n = len(self._chunks)
        self._avgdl = (sum(self._len) / n) if n else 0.0
        self._idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self._postings.items()}

    def search(self, query: str, k: int) -> list[Hit]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(self.tokenizer(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i in self._postings[term]:
                tf = self._tf[i][term]
                norm = 1 - self.b + self.b * (self._len[i] / self._avgdl if self._avgdl else 0.0)
                scores[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [Hit(self._chunks[i], s) for i, s in ranked]


EmbedFn = Callable[[list[str]], list[list[float]]]


class EmbeddingRetriever:
    """Dense retrieval with any embedding function you provide.

    ``embed`` takes a list of strings and returns one vector per string. It is called
    for chunks at index time and for each query. Use this to compare embedding models
    on the same corpus, chunking and dataset.
    """

    def __init__(self, embed: EmbedFn, name: str = "embedding", batch_size: int = 32, query_prefix: str = "", passage_prefix: str = ""):
        self.embed = embed
        self.name = name
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self._chunks: list[Chunk] = []
        self._vecs: list[list[float]] = []

    @staticmethod
    def _unit(v: Sequence[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v] if norm else list(v)

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        self._vecs = []
        texts = [self.passage_prefix + c.text for c in self._chunks]
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vecs = self.embed(batch)
            if len(vecs) != len(batch):
                raise ValueError(f"embed() returned {len(vecs)} vectors for {len(batch)} texts")
            self._vecs.extend(self._unit(v) for v in vecs)

    def search(self, query: str, k: int) -> list[Hit]:
        if not self._chunks:
            return []
        (qv,) = self.embed([self.query_prefix + query])
        q = self._unit(qv)
        scored = [(sum(a * b for a, b in zip(q, v)), i) for i, v in enumerate(self._vecs)]
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [Hit(self._chunks[i], s) for s, i in scored[:k]]


class RRFRetriever:
    """Reciprocal Rank Fusion of several retrievers (e.g. BM25 + embeddings)."""

    def __init__(self, retrievers: Sequence[Retriever], k_rrf: int = 60, depth: int = 50, name: str | None = None):
        if not retrievers:
            raise ValueError("RRFRetriever needs at least one retriever")
        self.retrievers = list(retrievers)
        self.k_rrf = k_rrf
        self.depth = depth
        self.name = name or "rrf(" + "+".join(r.name for r in self.retrievers) + ")"

    def index(self, chunks: Sequence[Chunk]) -> None:
        for r in self.retrievers:
            r.index(chunks)

    def search(self, query: str, k: int) -> list[Hit]:
        fused: dict[str, float] = defaultdict(float)
        by_id: dict[str, Chunk] = {}
        for r in self.retrievers:
            for rank, hit in enumerate(r.search(query, max(k, self.depth)), start=1):
                fused[hit.chunk.id] += 1.0 / (self.k_rrf + rank)
                by_id[hit.chunk.id] = hit.chunk
        order = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [Hit(by_id[cid], s) for cid, s in order]


def build_retriever(spec: str) -> Retriever:
    """Create a retriever from a CLI spec.

    ``bm25`` (word tokens), ``bm25-char3`` (character trigrams; any n >= 2),
    ``bm25-raw`` (no Persian normalization, ablation), or ``module:factory`` for a
    custom retriever (a zero-argument callable returning an object with ``name``,
    ``index`` and ``search``). The current directory is importable.
    """
    spec = spec.strip()
    if spec == "bm25":
        return BM25Retriever()
    if spec == "bm25-raw":
        return BM25Retriever(Tokenizer(normalize=False))
    if spec.startswith("bm25-char"):
        digits = spec[len("bm25-char") :]
        if not digits.isdigit():
            raise ValueError(f"invalid retriever spec {spec!r}; use bm25-char3 for trigrams")
        return BM25Retriever(Tokenizer(mode="char", ngram=int(digits)))
    if ":" in spec:
        module_name, _, attr = spec.partition(":")
        cwd = os.getcwd()
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
        try:
            factory = getattr(importlib.import_module(module_name), attr)
        except (ImportError, AttributeError) as exc:
            raise ValueError(f"cannot load retriever {spec!r}: {exc}") from exc
        retriever = factory()
        for attribute in ("name", "index", "search"):
            if not hasattr(retriever, attribute):
                raise ValueError(f"{spec!r} must return an object with 'name', 'index' and 'search' (missing {attribute!r})")
        return retriever
    raise ValueError(f"unknown retriever {spec!r}; use bm25, bm25-char3, bm25-raw or module:factory")
