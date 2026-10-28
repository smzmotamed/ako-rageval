"""Loading and validating corpus, dataset and prediction files (JSON Lines, UTF-8)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence


class DataError(ValueError):
    """A data file is malformed. The message includes the file and line number."""


@dataclass(frozen=True)
class Doc:
    id: str
    text: str
    title: str | None = None


@dataclass(frozen=True)
class Query:
    id: str
    question: str
    relevant: tuple[str, ...]
    reference_answer: str | None = None


@dataclass(frozen=True)
class Prediction:
    id: str
    answer: str
    contexts: tuple[str, ...] = ()


def _read(path: Any) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise DataError(f"{path}: file not found") from None
    except UnicodeDecodeError as exc:
        raise DataError(f"{path}: not valid UTF-8 ({exc.reason})") from None


def read_jsonl(path: Any) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line_number, object)`` for every non-blank line."""
    path = Path(path) if isinstance(path, str) else path
    for lineno, line in enumerate(_read(path).splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from None
        if not isinstance(obj, dict):
            raise DataError(f"{path}:{lineno}: each line must be a JSON object")
        yield lineno, obj


def _string(obj: dict[str, Any], key: str, where: str, *, required: bool = True) -> str | None:
    value = obj.get(key)
    if value is None:
        if required:
            raise DataError(f"{where}: missing required field {key!r}")
        return None
    if not isinstance(value, str) or (required and not value.strip()):
        raise DataError(f"{where}: field {key!r} must be a non-empty string")
    return value


def _unique(seen: set[str], ident: str, where: str, what: str) -> None:
    if ident in seen:
        raise DataError(f"{where}: duplicate {what} id {ident!r}")
    seen.add(ident)


def load_corpus(path: Any) -> list[Doc]:
    """JSONL with ``{"id", "text", "title"?}`` per line."""
    docs: list[Doc] = []
    seen: set[str] = set()
    for lineno, obj in read_jsonl(path):
        where = f"{path}:{lineno}"
        ident = _string(obj, "id", where)
        text = _string(obj, "text", where)
        assert ident is not None and text is not None
        _unique(seen, ident, where, "document")
        docs.append(Doc(ident, text, _string(obj, "title", where, required=False)))
    if not docs:
        raise DataError(f"{path}: corpus is empty")
    return docs


def load_dataset(path: Any, corpus: Sequence[Doc] | None = None) -> list[Query]:
    """JSONL with ``{"id", "question", "relevant": [doc ids], "reference_answer"?}`` per line."""
    known = {d.id for d in corpus} if corpus is not None else None
    queries: list[Query] = []
    seen: set[str] = set()
    for lineno, obj in read_jsonl(path):
        where = f"{path}:{lineno}"
        ident = _string(obj, "id", where)
        question = _string(obj, "question", where)
        assert ident is not None and question is not None
        _unique(seen, ident, where, "query")
        relevant = obj.get("relevant")
        if not isinstance(relevant, list) or not relevant or not all(isinstance(r, str) and r for r in relevant):
            raise DataError(f"{where}: field 'relevant' must be a non-empty list of document ids")
        if known is not None:
            missing = [r for r in relevant if r not in known]
            if missing:
                raise DataError(f"{where}: relevant document id(s) not in the corpus: {', '.join(missing)}")
        queries.append(Query(ident, question, tuple(dict.fromkeys(relevant)), _string(obj, "reference_answer", where, required=False)))
    if not queries:
        raise DataError(f"{path}: dataset is empty")
    return queries


def load_predictions(path: Any) -> list[Prediction]:
    """JSONL with ``{"id", "answer", "contexts"?: [str]}`` per line (output of your RAG system)."""
    out: list[Prediction] = []
    seen: set[str] = set()
    for lineno, obj in read_jsonl(path):
        where = f"{path}:{lineno}"
        ident = _string(obj, "id", where)
        answer = obj.get("answer")
        if not isinstance(answer, str):
            raise DataError(f"{where}: field 'answer' must be a string")
        assert ident is not None
        _unique(seen, ident, where, "prediction")
        contexts = obj.get("contexts", [])
        if not isinstance(contexts, list) or not all(isinstance(c, str) for c in contexts):
            raise DataError(f"{where}: field 'contexts' must be a list of strings")
        out.append(Prediction(ident, answer, tuple(contexts)))
    if not out:
        raise DataError(f"{path}: predictions file is empty")
    return out
