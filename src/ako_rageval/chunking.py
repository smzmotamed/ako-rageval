"""Chunking strategies. Chunks keep the *original* text; normalization happens at index time."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    index: int
    text: str

    @property
    def id(self) -> str:
        return f"{self.doc_id}#{self.index}"


class Chunker(Protocol):
    name: str

    def split(self, doc_id: str, text: str) -> list[Chunk]: ...


def _make(doc_id: str, pieces: list[str]) -> list[Chunk]:
    pieces = [p.strip() for p in pieces if p and p.strip()]
    return [Chunk(doc_id, i, p) for i, p in enumerate(pieces)]


@dataclass(frozen=True)
class WholeDocChunker:
    """No chunking: one chunk per document (baseline)."""

    name: str = "doc"

    def split(self, doc_id: str, text: str) -> list[Chunk]:
        return _make(doc_id, [re.sub(r"\s+", " ", text)])


@dataclass(frozen=True)
class FixedSizeChunker:
    """Sliding window over whitespace-separated words."""

    size: int = 100
    overlap: int = 20

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError("size must be >= 1")
        if not 0 <= self.overlap < self.size:
            raise ValueError("overlap must satisfy 0 <= overlap < size")

    @property
    def name(self) -> str:
        return f"fixed:{self.size}:{self.overlap}"

    def split(self, doc_id: str, text: str) -> list[Chunk]:
        tokens = text.split()
        if not tokens:
            return []
        step = self.size - self.overlap
        pieces = []
        for start in range(0, len(tokens), step):
            pieces.append(" ".join(tokens[start : start + self.size]))
            if start + self.size >= len(tokens):
                break
        return _make(doc_id, pieces)


# A sentence ends at . ! ? ؟ … followed by whitespace. "3.14" has no whitespace, so it is not split.
_SENTENCE_END = re.compile(r"(?<=[.!?؟…])\s+")


def split_sentences(text: str) -> list[str]:
    """Persian-aware sentence splitter (handles ؟ and …). Paragraph breaks also end a sentence."""
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", para).strip()
        if para:
            out.extend(s for s in _SENTENCE_END.split(para) if s)
    return out


@dataclass(frozen=True)
class SentenceChunker:
    """Groups ``sentences`` consecutive sentences, advancing by ``sentences - overlap``."""

    sentences: int = 3
    overlap: int = 0

    def __post_init__(self) -> None:
        if self.sentences < 1:
            raise ValueError("sentences must be >= 1")
        if not 0 <= self.overlap < self.sentences:
            raise ValueError("overlap must satisfy 0 <= overlap < sentences")

    @property
    def name(self) -> str:
        return f"sentence:{self.sentences}" + (f":{self.overlap}" if self.overlap else "")

    def split(self, doc_id: str, text: str) -> list[Chunk]:
        sents = split_sentences(text)
        if not sents:
            return []
        step = self.sentences - self.overlap
        pieces = []
        for start in range(0, len(sents), step):
            pieces.append(" ".join(sents[start : start + self.sentences]))
            if start + self.sentences >= len(sents):
                break
        return _make(doc_id, pieces)


@dataclass(frozen=True)
class ParagraphChunker:
    """One chunk per paragraph (blank-line separated); paragraphs shorter than ``min_words`` merge into the next."""

    min_words: int = 0
    name: str = "paragraph"

    def split(self, doc_id: str, text: str) -> list[Chunk]:
        paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
        pieces: list[str] = []
        carry = ""
        for p in paragraphs:
            if not p:
                continue
            candidate = f"{carry} {p}".strip() if carry else p
            if len(candidate.split()) < self.min_words:
                carry = candidate
            else:
                pieces.append(candidate)
                carry = ""
        if carry:
            if pieces:
                pieces[-1] = f"{pieces[-1]} {carry}"
            else:
                pieces.append(carry)
        return _make(doc_id, pieces)


def parse_chunker(spec: str) -> Chunker:
    """Build a chunker from a spec string.

    ``doc`` | ``paragraph[:min_words]`` | ``sentence[:n[:overlap]]`` | ``fixed[:size[:overlap]]``
    """
    parts = spec.strip().split(":")
    kind, args = parts[0].lower(), parts[1:]
    try:
        nums = [int(a) for a in args]
    except ValueError:
        raise ValueError(f"invalid chunker spec {spec!r}: arguments must be integers") from None
    if kind == "doc" and not nums:
        return WholeDocChunker()
    if kind == "paragraph" and len(nums) <= 1:
        return ParagraphChunker(*nums)
    if kind == "sentence" and len(nums) <= 2:
        return SentenceChunker(*nums)
    if kind == "fixed" and len(nums) <= 2:
        return FixedSizeChunker(*nums)
    raise ValueError(f"invalid chunker spec {spec!r}; expected doc | paragraph[:min] | sentence[:n[:overlap]] | fixed[:size[:overlap]]")
