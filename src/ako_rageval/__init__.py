"""ako-rageval: evaluation harness for Persian (Farsi) RAG pipelines."""

__version__ = "0.1.1"

from .chunking import Chunk, Chunker, FixedSizeChunker, ParagraphChunker, SentenceChunker, WholeDocChunker, parse_chunker, split_sentences
from .data import DataError, Doc, Prediction, Query, load_corpus, load_dataset, load_predictions
from .evaluate import DEFAULT_KS, AnswerResult, GenerationResult, QueryResult, RunResult, compare, evaluate_answers, evaluate_retrieval
from .metrics import Judge, hit_at_k, lexical_groundedness, mrr, ndcg_at_k, precision_at_k, rank_docs, recall_at_k, token_f1, token_recall
from .normalize import normalize
from .report import render_html, to_json
from .retrievers import BM25Retriever, EmbeddingRetriever, Hit, Retriever, RRFRetriever, build_retriever
from .tokenizer import Tokenizer

__all__ = [
    "__version__",
    "AnswerResult",
    "BM25Retriever",
    "Chunk",
    "Chunker",
    "DEFAULT_KS",
    "DataError",
    "Doc",
    "EmbeddingRetriever",
    "FixedSizeChunker",
    "GenerationResult",
    "Hit",
    "Judge",
    "ParagraphChunker",
    "Prediction",
    "Query",
    "QueryResult",
    "RRFRetriever",
    "Retriever",
    "RunResult",
    "SentenceChunker",
    "Tokenizer",
    "WholeDocChunker",
    "build_retriever",
    "compare",
    "evaluate_answers",
    "evaluate_retrieval",
    "hit_at_k",
    "lexical_groundedness",
    "load_corpus",
    "load_dataset",
    "load_predictions",
    "mrr",
    "ndcg_at_k",
    "normalize",
    "parse_chunker",
    "precision_at_k",
    "rank_docs",
    "recall_at_k",
    "render_html",
    "split_sentences",
    "to_json",
    "token_f1",
    "token_recall",
]
