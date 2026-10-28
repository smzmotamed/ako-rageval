# ako-rageval

[![CI](https://github.com/smzmotamed/ako-rageval/actions/workflows/ci.yml/badge.svg)](https://github.com/smzmotamed/ako-rageval/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Evaluation harness for Persian (Farsi) RAG pipelines.** Compare chunking strategies, retrievers and embedding models on your own corpus, track answer quality, and fail CI when retrieval regresses. فارسی: [README.fa.md](README.fa.md)

Persian text has problems that generic RAG tooling ignores: Arabic vs Persian `ي`/`ک`, Persian vs Arabic-Indic vs ASCII digits, ZWNJ (`می‌روم` / `می روم` / `میروم`), diacritics, `؟` and `«»`. This harness normalizes them for matching, evaluates retrieval at the **document level** so chunkers are comparable, and produces a self-contained HTML report (English or Persian/RTL).

- **Zero runtime dependencies**, Python 3.10+, pure-Python BM25.
- **Persian normalization** (`ako_rageval.normalize`) with explicit ZWNJ and digit policies.
- **Chunkers:** `doc`, `paragraph`, `sentence[:n[:overlap]]` (Persian-aware splitter), `fixed[:words[:overlap]]`.
- **Retrievers:** BM25 on words, BM25 on character n-grams (robust to morphology), a raw ablation without normalization, an **embedding retriever for any embedding function**, reciprocal rank fusion, or your own `module:factory`.
- **Metrics:** hit@k, recall@k, precision@k, nDCG@k, MRR; answer baselines `token_f1`, `token_recall`, `groundedness`; plug in an LLM judge.
- **CI gate:** `--fail-under recall@5=0.9` exits non-zero when quality drops.

## Quick start

```bash
pip install git+https://github.com/smzmotamed/ako-rageval   # PyPI release not published yet

ako-rageval compare --sample                       # bundled Persian corpus (23 docs, 27 questions)
ako-rageval compare --sample --html report.html --report-lang fa
ako-rageval eval-answers --sample
ako-rageval normalize "بزرگترين درياچه جهان ۱۲۳ كتابهاى"   # -> بزرگترین دریاچه جهان 123 کتابهای
```

Example (`compare --sample`, top and bottom rows):

```
config                       chunks        hit@5     recall@5  precision@5          mrr       ndcg@5
fixed:30:10 | bm25-char3         33        1.000        1.000        0.222        1.000        1.000
fixed:30:10 | bm25-word          33        1.000        1.000        0.222        1.000        0.997
doc | bm25-char3                 23        1.000        1.000        0.222        1.000        0.995
...
sentence:1 | bm25-word-raw       60        1.000        0.981        0.215        0.975        0.961
sentence:1 | bm25-char3          60        0.963        0.944        0.207        0.969        0.946
```

> The bundled sample is small and easy. It exists to demonstrate the tooling and as a smoke test, **not as a benchmark**. Build a golden dataset from your own documents and real user questions before drawing conclusions.

## Your data

Three UTF-8 JSON Lines files. Errors report the file and line number.

```jsonc
// corpus.jsonl
{"id": "d01", "title": "دماوند", "text": "قله‌ی دماوند با ارتفاع ..."}
// dataset.jsonl: relevant = ids of every document that answers the question
{"id": "q01", "question": "بلندترین قله‌ی ایران کدام است؟", "relevant": ["d01"], "reference_answer": "دماوند"}
// predictions.jsonl: output of YOUR rag system, contexts = the passages it retrieved
{"id": "q01", "answer": "دماوند", "contexts": ["قله‌ی دماوند با ارتفاع ..."]}
```

```bash
ako-rageval run --corpus corpus.jsonl --dataset dataset.jsonl --chunker sentence:3 --retriever bm25 --k 1,3,5,10
ako-rageval compare --corpus corpus.jsonl --dataset dataset.jsonl \
    --chunkers doc,sentence:2,sentence:3,fixed:100:20 --retrievers bm25,bm25-char3 --html report.html
ako-rageval eval-answers --dataset dataset.jsonl --predictions predictions.jsonl --fail-under groundedness=0.8
ako-rageval sample --out my-data      # export the sample files as a template
```

Exit codes: `0` ok, `1` a `--fail-under` threshold was missed, `2` bad input or usage.

## Compare embedding models

The CLI cannot call your embedding provider, so use the Python API. Any `embed(list[str]) -> list[vector]` works (sentence-transformers, an HTTP API, ...).

```python
from ako_rageval import (EmbeddingRetriever, BM25Retriever, RRFRetriever, SentenceChunker,
                         compare, load_corpus, load_dataset, render_html)

corpus = load_corpus("corpus.jsonl")
dataset = load_dataset("dataset.jsonl", corpus)

def make_embedder(model_name):
    from sentence_transformers import SentenceTransformer       # your dependency, not ours
    model = SentenceTransformer(model_name)
    return lambda texts: model.encode(texts, normalize_embeddings=True).tolist()

retrievers = [
    BM25Retriever,                                                        # a class or factory is fine
    lambda: EmbeddingRetriever(make_embedder("model-a"), name="model-a"),
    lambda: EmbeddingRetriever(make_embedder("model-b"), name="model-b"),
    lambda: RRFRetriever([BM25Retriever(), EmbeddingRetriever(make_embedder("model-a"), name="model-a")]),
]
runs = compare(corpus, dataset, [SentenceChunker(2), SentenceChunker(3)], retrievers, ks=(1, 3, 5, 10))
open("report.html", "w", encoding="utf-8").write(render_html(runs, lang="fa"))
```

Some embedding models need prefixes for queries and passages: pass `query_prefix=` / `passage_prefix=` to `EmbeddingRetriever`. To use a custom retriever from the CLI, expose `module:factory` (a zero-argument callable returning an object with `name`, `index(chunks)` and `search(query, k)`):

```bash
ako-rageval compare --corpus c.jsonl --dataset d.jsonl --retrievers bm25,my_retrievers:make_hybrid
```

## How evaluation works

1. Each document is split by the chunker (original text is kept; normalization only happens when indexing and comparing).
2. A retriever indexes the chunks and returns a ranked list for every question.
3. Chunks are mapped to their source documents and de-duplicated, so a document counts once at its best rank. This is what makes `doc`, `sentence:1` and `fixed:100` comparable.
4. Metrics are averaged over questions. Runs are ranked by nDCG at k=5 (or the largest k if 5 is not requested), then MRR.

`precision@k` divides by `k`, so it is low whenever a question has fewer than `k` relevant documents. Prefer `recall@k`, `ndcg@k` and `mrr` for ranking.

### Answer metrics are baselines

`token_f1`, `token_recall` and `groundedness` are lexical. They are deterministic and cheap, which makes them good for regression tracking, but they cannot judge meaning:

- `token_f1` punishes verbose answers (a full sentence scores low against a one-word reference). Use `token_recall` when references are short.
- `groundedness` is the share of answer words that occur in the retrieved contexts. Low values flag likely hallucinations, but a high value does not prove the answer is right, and words echoed from the question can inflate it.

For semantic judgement implement the `Judge` protocol (`name`, `score(question, answer, contexts, reference) -> float in [0, 1]`) and pass it to `evaluate_answers(..., judge=...)`.

## Normalization

```python
from ako_rageval import normalize
normalize("می‌روم")                    # 'می روم'   (zwnj="space", default)
normalize("می‌روم", zwnj="keep")       # keeps the ZWNJ
normalize("۱۲۳ ٤٥٦", digits="persian")      # '۱۲۳ ۴۵۶'
normalize("آب", fold_alef_madda=True)       # 'اب'  (off by default: it conflates words)
```

Default steps: NFKC, Arabic to Persian letters (`ي ى ك ة ۀ أ إ ٱ`), diacritics and tatweel removal, invisible/bidi marks removal, ZWNJ policy, digits to ASCII, Arabic punctuation and quotes to ASCII, lowercase, whitespace collapse. Plurals and glued compounds (`بزرگترین` vs `بزرگ‌ترین`) are not unified; character n-gram BM25 (`bm25-char3`) is the cheap mitigation, and the comparison table shows whether it helps on your data.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

[Apache-2.0](LICENSE) © Akomind
