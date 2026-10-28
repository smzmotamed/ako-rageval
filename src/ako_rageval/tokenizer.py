"""Tokenization for lexical retrieval and lexical answer metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .normalize import normalize

# Frequent Persian function words (after normalization). Deliberately small and
# conservative: removing too much hurts short questions more than it helps.
STOPWORDS_FA: frozenset[str] = frozenset(
    normalize(w)
    for w in """
    و در به از که این آن را با است بود شد شده برای یک هم تا یا بر اما هر نیز می خود
    باشد بعد ها های ای تر ترین چه کدام کیست چیست کجاست چند چقدر آیا
    """.split()
)

_WORD = re.compile(r"[^\W_]+", re.UNICODE)

TokenizeFn = Callable[[str], list[str]]


def words(text: str) -> list[str]:
    """Split *already normalized* text into word tokens (letters and digits)."""
    return _WORD.findall(text)


def char_ngrams(token: str, n: int = 3) -> list[str]:
    """Character n-grams of ``#token#`` (boundary-padded). Robust to Persian morphology."""
    padded = f"#{token}#"
    if len(padded) <= n:
        return [padded]
    return [padded[i : i + n] for i in range(len(padded) - n + 1)]


@dataclass(frozen=True)
class Tokenizer:
    """Configurable tokenizer used by BM25 and the lexical answer metrics.

    ``mode="word"`` yields words; ``mode="char"`` yields boundary-padded character
    n-grams of each word. ``normalize=False`` disables Persian normalization (useful
    as an ablation to measure what normalization buys you).
    """

    mode: str = "word"
    ngram: int = 3
    normalize: bool = True
    stopwords: frozenset[str] = STOPWORDS_FA

    def __post_init__(self) -> None:
        if self.mode not in ("word", "char"):
            raise ValueError(f"mode must be 'word' or 'char', got {self.mode!r}")
        if self.ngram < 2:
            raise ValueError("ngram must be >= 2")

    @property
    def name(self) -> str:
        base = "word" if self.mode == "word" else f"char{self.ngram}"
        return base if self.normalize else f"{base}-raw"

    def __call__(self, text: str) -> list[str]:
        text = normalize(text) if self.normalize else text.lower()
        toks = [t for t in words(text) if t not in self.stopwords]
        if self.mode == "word":
            return toks
        out: list[str] = []
        for t in toks:
            out.extend(char_ngrams(t, self.ngram))
        return out


def content_tokens(text: str) -> list[str]:
    """Normalized word tokens without stopwords (used by answer metrics)."""
    return Tokenizer()(text)
