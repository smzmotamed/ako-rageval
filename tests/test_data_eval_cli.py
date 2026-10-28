import json
from importlib import resources

import pytest

from ako_rageval import (
    BM25Retriever,
    DataError,
    FixedSizeChunker,
    SentenceChunker,
    WholeDocChunker,
    compare,
    evaluate_answers,
    evaluate_retrieval,
    load_corpus,
    load_dataset,
    load_predictions,
    render_html,
    to_json,
)
from ako_rageval.cli import main


def sample(name):
    return resources.files("ako_rageval").joinpath("sample", name)


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(sample("corpus.jsonl"))


@pytest.fixture(scope="module")
def dataset(corpus):
    return load_dataset(sample("dataset.jsonl"), corpus)


class TestData:
    def test_sample_files_are_valid(self, corpus, dataset):
        assert len(corpus) == 23 and len(dataset) == 27
        preds = load_predictions(sample("predictions.jsonl"))
        assert {p.id for p in preds} <= {q.id for q in dataset}

    def test_errors_include_file_and_line(self, tmp_path):
        bad = tmp_path / "c.jsonl"
        bad.write_text('{"id": "a", "text": "x"}\nnot json\n', encoding="utf-8")
        with pytest.raises(DataError, match=r"c\.jsonl:2: invalid JSON"):
            load_corpus(bad)
        with pytest.raises(DataError, match="file not found"):
            load_corpus(tmp_path / "missing.jsonl")

    def test_corpus_validation(self, tmp_path):
        with pytest.raises(DataError, match="missing required field 'text'"):
            load_corpus(write_jsonl(tmp_path / "a.jsonl", [{"id": "a"}]))
        with pytest.raises(DataError, match="duplicate document id 'a'"):
            load_corpus(write_jsonl(tmp_path / "b.jsonl", [{"id": "a", "text": "x"}, {"id": "a", "text": "y"}]))
        with pytest.raises(DataError, match="empty"):
            load_corpus(write_jsonl(tmp_path / "c.jsonl", []))

    def test_dataset_validation(self, tmp_path, corpus):
        with pytest.raises(DataError, match="non-empty list"):
            load_dataset(write_jsonl(tmp_path / "a.jsonl", [{"id": "q", "question": "?", "relevant": []}]))
        with pytest.raises(DataError, match="not in the corpus: nope"):
            load_dataset(write_jsonl(tmp_path / "b.jsonl", [{"id": "q", "question": "?", "relevant": ["nope"]}]), corpus)

    def test_non_utf8_is_reported(self, tmp_path):
        p = tmp_path / "x.jsonl"
        p.write_bytes(b"\xff\xfe\x00")
        with pytest.raises(DataError, match="not valid UTF-8"):
            load_corpus(p)


class TestEvaluation:
    def test_bm25_on_sample_is_strong(self, corpus, dataset):
        run = evaluate_retrieval(corpus, dataset, SentenceChunker(3), BM25Retriever())
        assert run.summary["recall@5"] == 1.0
        assert run.summary["ndcg@5"] > 0.95
        assert run.failures() == []
        assert run.label == "sentence:3 | bm25-word"
        assert run.primary_k == 5

    def test_normalization_matters_on_noisy_queries(self, corpus, dataset):
        from ako_rageval import Tokenizer

        noisy = [q for q in dataset if q.id in {"q24", "q25"}]  # Arabic-keyboard letters, ASCII digits
        good = evaluate_retrieval(corpus, noisy, WholeDocChunker(), BM25Retriever())
        raw = evaluate_retrieval(corpus, noisy, WholeDocChunker(), BM25Retriever(Tokenizer(normalize=False)))
        assert good.summary["mrr"] >= raw.summary["mrr"]
        assert good.summary["recall@5"] == 1.0

    def test_compare_sorts_best_first_and_is_deterministic(self, corpus, dataset):
        chunkers = [WholeDocChunker(), SentenceChunker(1), FixedSizeChunker(30, 10)]
        runs = compare(corpus, dataset, chunkers, [BM25Retriever, lambda: BM25Retriever(name="b2")])
        assert len(runs) == 6
        scores = [r.score for r in runs]
        assert scores == sorted(scores, reverse=True)
        again = compare(corpus, dataset, chunkers, [BM25Retriever, lambda: BM25Retriever(name="b2")])
        assert [r.label for r in again] == [r.label for r in runs]

    def test_custom_ks(self, corpus, dataset):
        run = evaluate_retrieval(corpus, dataset, WholeDocChunker(), BM25Retriever(), ks=[2, 1])
        assert run.ks == (1, 2) and run.primary_k == 2
        assert {"hit@1", "hit@2", "recall@2", "ndcg@1", "mrr"} <= run.summary.keys()
        with pytest.raises(ValueError):
            evaluate_retrieval(corpus, dataset, WholeDocChunker(), BM25Retriever(), ks=[0])

    def test_chunk_level_dedup_gives_document_level_metrics(self, corpus, dataset):
        run = evaluate_retrieval(corpus, dataset, SentenceChunker(1), BM25Retriever())
        for q in run.queries:
            assert len(q.retrieved) == len(set(q.retrieved))

    def test_answers(self, dataset):
        preds = load_predictions(sample("predictions.jsonl"))
        with pytest.raises(DataError, match="no prediction for"):
            evaluate_answers(dataset, preds)
        res = evaluate_answers(dataset, preds, allow_missing=True)
        by_id = {a.id: a for a in res.queries}
        assert res.summary["answered"] == 6 and res.summary["coverage"] == pytest.approx(6 / 27)
        assert by_id["q02"].groundedness == 1.0
        assert by_id["q04"].groundedness < 0.7  # hallucinated answer uses words absent from the context
        assert by_id["q08"].token_f1 < by_id["q05"].token_f1
        assert by_id["q02"].token_recall == 1.0 and by_id["q08"].token_recall == 0.0
        assert "token_recall" in res.summary
        assert len(res.missing) == 21

    def test_answers_unknown_id_and_judge(self, dataset):
        from ako_rageval import Prediction

        with pytest.raises(DataError, match="unknown query id"):
            evaluate_answers(dataset, [Prediction("zzz", "x")], allow_missing=True)

        class Half:
            name = "half"

            def score(self, question, answer, contexts, reference):
                return 0.5

        res = evaluate_answers(dataset, [Prediction("q05", "نوروز")], judge=Half(), allow_missing=True)
        assert res.summary["judge:half"] == 0.5


class TestReport:
    def test_html_is_escaped_and_localised(self, corpus, dataset):
        q = dataset[0]
        evil = type(q)(q.id, "<script>alert(1)</script> قله؟", q.relevant, q.reference_answer)
        runs = compare(corpus, [evil, *dataset[1:]], [WholeDocChunker()], [BM25Retriever])
        html_en = render_html(runs, version="9.9")
        assert "<script>alert(1)</script>" not in html_en and "&lt;script&gt;" in html_en
        assert "dir='ltr'" in html_en and "ako-rageval 9.9" in html_en
        assert "<script" not in html_en.replace("&lt;script&gt;", "")
        html_fa = render_html(runs, lang="fa")
        assert "dir='rtl'" in html_fa and "گزارش ارزیابی RAG" in html_fa
        with pytest.raises(ValueError):
            render_html(runs, lang="de")

    def test_report_with_failures_and_generation(self, corpus, dataset):
        # a retriever that finds nothing -> every query is a failure
        class Empty:
            name = "empty"

            def index(self, chunks):
                pass

            def search(self, query, k):
                return []

        runs = compare(corpus, dataset, [WholeDocChunker()], [Empty])
        assert len(runs[0].failures()) == len(dataset)
        gen = evaluate_answers(dataset, load_predictions(sample("predictions.jsonl")), allow_missing=True)
        html = render_html(runs, gen)
        assert html.count("class='bad'") >= len(dataset)
        assert "groundedness" in html

    def test_json_roundtrip(self, corpus, dataset):
        runs = compare(corpus, dataset, [WholeDocChunker()], [BM25Retriever])
        data = json.loads(to_json(runs))
        assert data["runs"][0]["label"] == "doc | bm25-word"
        assert data["runs"][0]["summary"]["recall@5"] == runs[0].summary["recall@5"]
        assert "\\u" not in to_json(runs)  # Persian text is written as UTF-8, not escaped


class TestCli:
    def test_normalize(self, capsys):
        assert main(["normalize", "كتاب ۱۲۳"]) == 0
        assert capsys.readouterr().out.strip() == "کتاب 123"

    def test_run_and_outputs(self, tmp_path, capsys):
        j, h = tmp_path / "r.json", tmp_path / "r.html"
        code = main(["run", "--sample", "--chunker", "sentence:3", "--retriever", "bm25", "--json", str(j), "--html", str(h), "--report-lang", "fa"])
        out = capsys.readouterr().out
        assert code == 0
        assert "23 documents, 27 queries" in out and "sentence:3 | bm25-word" in out
        assert json.loads(j.read_text(encoding="utf-8"))["runs"][0]["chunker"] == "sentence:3"
        assert "dir='rtl'" in h.read_text(encoding="utf-8")

    def test_compare_prints_ranked_table(self, capsys):
        assert main(["compare", "--sample", "--chunkers", "doc,sentence:1", "--retrievers", "bm25,bm25-char3"]) == 0
        lines = [ln for ln in capsys.readouterr().out.splitlines() if "|" in ln]
        assert len(lines) == 4

    def test_fail_under_gate(self, capsys):
        assert main(["run", "--sample", "--fail-under", "recall@5=0.5"]) == 0
        assert main(["run", "--sample", "--fail-under", "recall@5=1.01"]) == 1
        assert "FAIL" in capsys.readouterr().err
        assert main(["run", "--sample", "--fail-under", "bogus@5=0.5"]) == 2

    def test_user_files(self, tmp_path, corpus, capsys):
        c = write_jsonl(tmp_path / "c.jsonl", [{"id": "a", "text": "دماوند قله است"}, {"id": "b", "text": "سعدی شاعر است"}])
        d = write_jsonl(tmp_path / "d.jsonl", [{"id": "q", "question": "قله کدام است", "relevant": ["a"]}])
        assert main(["run", "--corpus", str(c), "--dataset", str(d), "--chunker", "doc", "--k", "1,2"]) == 0
        assert "hit@2" in capsys.readouterr().out

    def test_errors_exit_2(self, tmp_path, capsys):
        assert main(["run"]) == 2
        assert main(["run", "--sample", "--chunker", "bogus"]) == 2
        assert main(["run", "--sample", "--retriever", "nope"]) == 2
        assert main(["run", "--corpus", str(tmp_path / "no.jsonl"), "--dataset", str(tmp_path / "no2.jsonl")]) == 2
        err = capsys.readouterr().err
        assert err.count("ako-rageval:") == 4
        with pytest.raises(SystemExit) as exc:
            main(["run", "--sample", "--k", "a,b"])
        assert exc.value.code == 2

    def test_eval_answers(self, tmp_path, capsys):
        assert main(["eval-answers", "--sample"]) == 0
        out = capsys.readouterr().out
        assert "groundedness" in out and "q04" in out
        assert main(["eval-answers", "--sample", "--fail-under", "groundedness=0.99"]) == 1
        assert main(["eval-answers", "--dataset", str(tmp_path / "x"), "--predictions", str(tmp_path / "y")]) == 2

    def test_sample_command_writes_files(self, tmp_path, capsys):
        assert main(["sample", "--out", str(tmp_path / "s")]) == 0
        assert sorted(p.name for p in (tmp_path / "s").iterdir()) == ["corpus.jsonl", "dataset.jsonl", "predictions.jsonl"]
        # the exported files are usable as user data
        assert main(["run", "--corpus", str(tmp_path / "s/corpus.jsonl"), "--dataset", str(tmp_path / "s/dataset.jsonl")]) == 0
