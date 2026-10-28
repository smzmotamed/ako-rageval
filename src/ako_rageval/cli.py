"""Command line interface: ``ako-rageval``."""

from __future__ import annotations

import argparse
import os
import sys
from importlib import resources
from pathlib import Path
from typing import Sequence

from . import __version__
from .chunking import parse_chunker
from .data import DataError, load_corpus, load_dataset, load_predictions
from .evaluate import DEFAULT_KS, GenerationResult, RunResult, compare, evaluate_answers
from .normalize import normalize
from .report import render_html, to_json
from .retrievers import build_retriever

SAMPLE_FILES = ("corpus.jsonl", "dataset.jsonl", "predictions.jsonl")


def _sample(name: str):
    return resources.files("ako_rageval").joinpath("sample", name)


def _ks(value: str) -> tuple[int, ...]:
    try:
        ks = tuple(sorted({int(x) for x in value.split(",") if x.strip()}))
    except ValueError:
        raise argparse.ArgumentTypeError("--k must be a comma-separated list of integers, e.g. 1,3,5,10") from None
    if not ks or ks[0] < 1:
        raise argparse.ArgumentTypeError("--k values must be positive integers")
    return ks


def _threshold(value: str) -> tuple[str, float]:
    metric, sep, number = value.partition("=")
    try:
        if not sep or not metric:
            raise ValueError
        return metric.strip(), float(number)
    except ValueError:
        raise argparse.ArgumentTypeError("--fail-under expects METRIC=VALUE, e.g. recall@5=0.8") from None


def _common_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--corpus", help="corpus JSONL ({id, text, title?})")
    p.add_argument("--dataset", help="dataset JSONL ({id, question, relevant, reference_answer?})")
    p.add_argument("--sample", action="store_true", help="use the bundled Persian sample corpus and dataset")
    p.add_argument("--k", type=_ks, default=DEFAULT_KS, metavar="K,K,...", help="cutoffs (default: 1,3,5,10)")
    p.add_argument("--json", dest="json_out", metavar="FILE", help="write full results as JSON")
    p.add_argument("--html", dest="html_out", metavar="FILE", help="write a self-contained HTML report")
    p.add_argument("--report-lang", choices=("en", "fa"), default="en", help="language of the HTML report (default: en)")
    p.add_argument("--fail-under", action="append", type=_threshold, default=[], metavar="METRIC=VALUE", help="exit 1 if the best run's metric is below VALUE (repeatable), e.g. recall@5=0.8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ako-rageval", description="Evaluation harness for Persian RAG pipelines.")
    parser.add_argument("--version", action="version", version=f"ako-rageval {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    n = sub.add_parser("normalize", help="print Persian-normalized text")
    n.add_argument("text", help="text to normalize")
    n.add_argument("--zwnj", choices=("space", "keep", "remove"), default="space")
    n.add_argument("--digits", choices=("ascii", "persian", "keep"), default="ascii")
    n.add_argument("--fold-alef-madda", action="store_true", help="also map آ to ا")

    r = sub.add_parser("run", help="evaluate one chunker + retriever")
    _common_data_args(r)
    r.add_argument("--chunker", default="sentence:2", help="doc | paragraph[:min] | sentence[:n[:overlap]] | fixed[:size[:overlap]] (default: sentence:2)")
    r.add_argument("--retriever", default="bm25", help="bm25 | bm25-char3 | bm25-raw | module:factory (default: bm25)")

    c = sub.add_parser("compare", help="evaluate every chunker x retriever combination")
    _common_data_args(c)
    c.add_argument("--chunkers", default="doc,sentence:1,sentence:2,sentence:3,fixed:30:10", help="comma-separated chunker specs")
    c.add_argument("--retrievers", default="bm25,bm25-char3,bm25-raw", help="comma-separated retriever specs")

    a = sub.add_parser("eval-answers", help="score answers from your RAG system (lexical baselines)")
    a.add_argument("--dataset", help="dataset JSONL with reference_answer")
    a.add_argument("--predictions", help="predictions JSONL ({id, answer, contexts?})")
    a.add_argument("--sample", action="store_true", help="use the bundled sample dataset and predictions")
    a.add_argument("--allow-missing", action="store_true", help="skip dataset queries that have no prediction")
    a.add_argument("--json", dest="json_out", metavar="FILE")
    a.add_argument("--html", dest="html_out", metavar="FILE")
    a.add_argument("--report-lang", choices=("en", "fa"), default="en")
    a.add_argument("--fail-under", action="append", type=_threshold, default=[], metavar="METRIC=VALUE", help="e.g. groundedness=0.9")

    s = sub.add_parser("sample", help="write the bundled sample files to a directory")
    s.add_argument("--out", default="ako-rageval-sample", help="target directory (default: ako-rageval-sample)")
    return parser


def _load(args: argparse.Namespace):
    if args.sample:
        corpus = load_corpus(_sample("corpus.jsonl"))
        dataset = load_dataset(_sample("dataset.jsonl"), corpus)
    else:
        if not (args.corpus and args.dataset):
            raise DataError("provide --corpus and --dataset, or use --sample")
        corpus = load_corpus(Path(args.corpus))
        dataset = load_dataset(Path(args.dataset), corpus)
    return corpus, dataset


def _table(runs: Sequence[RunResult]) -> str:
    k = runs[0].primary_k
    cols = [f"hit@{k}", f"recall@{k}", f"precision@{k}", "mrr", f"ndcg@{k}"]
    width = max(len(r.label) for r in runs)
    lines = [f"{'config':<{width}}  chunks  " + "  ".join(f"{c:>11}" for c in cols)]
    for r in runs:
        row = [r.summary[f"hit@{r.primary_k}"], r.summary[f"recall@{r.primary_k}"], r.summary[f"precision@{r.primary_k}"], r.summary["mrr"], r.summary[f"ndcg@{r.primary_k}"]]
        lines.append(f"{r.label:<{width}}  {r.num_chunks:>6}  " + "  ".join(f"{v:>11.3f}" for v in row))
    return "\n".join(lines)


def _check_thresholds(summary: dict[str, float], thresholds: Sequence[tuple[str, float]]) -> list[str]:
    problems = []
    for metric, minimum in thresholds:
        if metric not in summary:
            raise DataError(f"--fail-under: unknown metric {metric!r}; available: {', '.join(sorted(summary))}")
        if summary[metric] < minimum:
            problems.append(f"{metric} = {summary[metric]:.3f} is below {minimum}")
    return problems


def _write(path: str, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")
    print(f"wrote {path}")


def _cmd_retrieval(args: argparse.Namespace) -> int:
    corpus, dataset = _load(args)
    if args.command == "run":
        chunker_specs, retriever_specs = [args.chunker], [args.retriever]
    else:
        chunker_specs = [s.strip() for s in args.chunkers.split(",") if s.strip()]
        retriever_specs = [s.strip() for s in args.retrievers.split(",") if s.strip()]
    try:
        chunkers = [parse_chunker(s) for s in chunker_specs]
        retrievers = [(lambda spec=spec: build_retriever(spec)) for spec in retriever_specs]
        for spec in retriever_specs:
            build_retriever(spec)  # fail fast on a bad spec before any evaluation work
    except ValueError as exc:
        raise DataError(str(exc)) from None
    runs = compare(corpus, dataset, chunkers, retrievers, args.k)
    print(f"{len(corpus)} documents, {len(dataset)} queries, ranked by nDCG@{runs[0].primary_k}\n")
    print(_table(runs))
    if args.json_out:
        _write(args.json_out, to_json(runs))
    if args.html_out:
        _write(args.html_out, render_html(runs, lang=args.report_lang, version=__version__))
    problems = _check_thresholds(runs[0].summary, args.fail_under)
    for p in problems:
        print(f"FAIL: best run {runs[0].label}: {p}", file=sys.stderr)
    return 1 if problems else 0


def _cmd_answers(args: argparse.Namespace) -> int:
    if args.sample:
        dataset = load_dataset(_sample("dataset.jsonl"))
        predictions = load_predictions(_sample("predictions.jsonl"))
        allow_missing = True
    else:
        if not (args.dataset and args.predictions):
            raise DataError("provide --dataset and --predictions, or use --sample")
        dataset = load_dataset(Path(args.dataset))
        predictions = load_predictions(Path(args.predictions))
        allow_missing = args.allow_missing
    result: GenerationResult = evaluate_answers(dataset, predictions, allow_missing=allow_missing)
    for key, value in result.summary.items():
        print(f"{key:<14} {value:.3f}" if key != "answered" else f"{key:<14} {int(value)}")
    print()
    for q in result.queries:
        f1 = "  -  " if q.token_f1 is None else f"{q.token_f1:.2f}"
        rc = "  -  " if q.token_recall is None else f"{q.token_recall:.2f}"
        gr = "  -  " if q.groundedness is None else f"{q.groundedness:.2f}"
        print(f"{q.id}  token_f1={f1}  token_recall={rc}  groundedness={gr}")
    if result.missing:
        print(f"\n{len(result.missing)} query id(s) without a prediction were skipped")
    if args.json_out:
        _write(args.json_out, to_json([], result))
    if args.html_out:
        _write(args.html_out, render_html([], result, lang=args.report_lang, version=__version__))
    problems = _check_thresholds(result.summary, args.fail_under)
    for p in problems:
        print(f"FAIL: {p}", file=sys.stderr)
    return 1 if problems else 0


def _cmd_sample(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in SAMPLE_FILES:
        (out / name).write_text(_sample(name).read_text(encoding="utf-8"), encoding="utf-8")
        print(f"wrote {out / name}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "normalize":
            print(normalize(args.text, zwnj=args.zwnj, digits=args.digits, fold_alef_madda=args.fold_alef_madda))
            return 0
        if args.command in ("run", "compare"):
            return _cmd_retrieval(args)
        if args.command == "eval-answers":
            return _cmd_answers(args)
        if args.command == "sample":
            return _cmd_sample(args)
    except DataError as exc:
        print(f"ako-rageval: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:  # e.g. `ako-rageval compare ... | head`: not an error
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
