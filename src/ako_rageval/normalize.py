"""Persian text normalization for retrieval and matching.

The goal is *matching*, not typography: two spellings a Persian reader considers
identical should normalize to the same string. Original text is never modified
by the harness; normalization is applied only when indexing and comparing.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

Zwnj = Literal["space", "keep", "remove"]
Digits = Literal["ascii", "persian", "keep"]

ZWNJ = "‌"

# Arabic letter variants -> the Persian letter users actually mean.
_LETTER_MAP = {
    "ي": "ی",  # ي Arabic yeh        -> ی
    "ى": "ی",  # ى alef maksura      -> ی
    "ے": "ی",  # ے yeh barree        -> ی
    "ك": "ک",  # ك Arabic kaf        -> ک
    "ة": "ه",  # ة teh marbuta       -> ه
    "ۀ": "ه",  # ۀ heh with yeh above-> ه
    "أ": "ا",  # أ                   -> ا
    "إ": "ا",  # إ                   -> ا
    "ٱ": "ا",  # ٱ                   -> ا
}

_PUNCT_MAP = {
    "،": ",",  # ، Arabic comma
    "؛": ";",  # ؛ Arabic semicolon
    "؟": "?",  # ؟ Arabic question mark
    "٪": "%",  # ٪ Arabic percent
    "٫": ".",  # ٫ Arabic decimal separator
    "٬": ",",  # ٬ Arabic thousands separator
    "«": '"',  # «
    "»": '"',  # »
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "–": "-",
    "—": "-",
}

_PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_TO_ASCII = {ord(c): str(i) for i, c in enumerate(_PERSIAN_DIGITS)}
_TO_ASCII.update({ord(c): str(i) for i, c in enumerate(_ARABIC_DIGITS)})
_TO_PERSIAN = {ord(str(i)): c for i, c in enumerate(_PERSIAN_DIGITS)}
_TO_PERSIAN.update({ord(c): _PERSIAN_DIGITS[i] for i, c in enumerate(_ARABIC_DIGITS)})

# Harakat (064B-065F), superscript alef (0670), tatweel (0640), Quranic marks (06D6-06ED).
_DIACRITICS = re.compile("[ً-ٰٟـۖ-ۭ]")
# Invisible direction marks and zero-width characters that are not ZWNJ.
_INVISIBLE = re.compile("[​‍‎‏‪-‮⁦-⁩﻿­]")
_WHITESPACE = re.compile(r"\s+")


def normalize(
    text: str,
    *,
    zwnj: Zwnj = "space",
    digits: Digits = "ascii",
    fold_alef_madda: bool = False,
    punctuation: bool = True,
    lowercase: bool = True,
) -> str:
    """Normalize Persian (and mixed Persian/Latin) text for matching.

    Steps: NFKC (folds Arabic presentation forms), Arabic->Persian letters,
    diacritics/tatweel removal, invisible/bidi marks removal, ZWNJ policy,
    digit policy, punctuation mapping, whitespace collapsing.

    * ``zwnj="space"`` (default) makes ``می‌روم`` and ``می روم`` identical; use
      ``"keep"`` to preserve it or ``"remove"`` to glue the parts together.
    * ``digits="ascii"`` maps Persian and Arabic-Indic digits to ``0-9``.
    * ``fold_alef_madda=True`` also maps ``آ`` to ``ا`` (off by default because
      it conflates words such as ``آب`` and ``اب``).
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(_LETTER_MAP.get(ch, ch) for ch in text)
    if fold_alef_madda:
        text = text.replace("آ", "ا")
    text = _DIACRITICS.sub("", text)
    text = _INVISIBLE.sub("", text)
    if zwnj == "space":
        text = text.replace(ZWNJ, " ")
    elif zwnj == "remove":
        text = text.replace(ZWNJ, "")
    elif zwnj != "keep":
        raise ValueError(f"zwnj must be 'space', 'keep' or 'remove', got {zwnj!r}")
    if digits == "ascii":
        text = text.translate(_TO_ASCII)
    elif digits == "persian":
        text = text.translate(_TO_PERSIAN)
    elif digits != "keep":
        raise ValueError(f"digits must be 'ascii', 'persian' or 'keep', got {digits!r}")
    if punctuation:
        text = "".join(_PUNCT_MAP.get(ch, ch) for ch in text)
    if lowercase:
        text = text.lower()
    return _WHITESPACE.sub(" ", text).strip()
