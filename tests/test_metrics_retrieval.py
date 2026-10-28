import math

import pytest

from ako_rageval import (
    BM25Retriever,
    Chunk,
    EmbeddingRetriever,
    Hit,
    RRFRetriever,
    build_retriever,
    hit_at_k,
    lexical_groundedness,
    mrr,
    ndcg_at_k,
    precision_at_k,
    rank_docs,
    recall_at_k,
    token_f1,
    token_recall,
)


class TestRetrievalMetrics:
    ranked = ["a", "b", "c", "d"]

    def test_hit_recall_precision(self):
        rel = {"b", "d", "z"}
        assert hit_at_k(self.ranked, rel, 1) == 0.0
        assert hit_at_k(self.ranked, rel, 2) == 1.0
        assert recall_at_k(self.ranked, rel, 2) == pytest.approx(1 / 3)
        assert recall_at_k(self.ranked, rel, 4) == pytest.approx(2 / 3)
        assert precision_at_k(self.ranked, rel, 4) == 0.5
        assert precision_at_k(["a"], rel, 4) == 0.0  # denominator is k even when fewer results exist

    def test_mrr(self):
        assert mrr(self.ranked, {"c"}) == pytest.approx(1 / 3)
        assert mrr(self.ranked, {"c"}, k=2) == 0.0
        assert mrr(self.ranked, {"zzz"}) == 0.0

    def test_ndcg_known_values(self):
        # relevant at ranks 1 and 3, two relevant in total
        dcg = 1 / math.log2(2) + 1 / math.log2(4)
        ideal = 1 / math.log2(2) + 1 / math.log2(3)
        assert ndcg_at_k(["a", "x", "b"], {"a", "b"}, 3) == pytest.approx(dcg / ideal)
        assert ndcg_at_k(["a", "b"], {"a", "b"}, 5) == pytest.approx(1.0)
        assert ndcg_at_k(["x"], {"a"}, 3) == 0.0

    def test_validation(self):
        with pytest.raises(ValueError):
            recall_at_k(["a"], set(), 1)
        with pytest.raises(ValueError):
            hit_at_k(["a"], {"a"}, 0)

    def test_rank_docs_dedupes_in_order(self):
        hits = [Hit(Chunk(d, i, "x"), 1.0) for i, d in enumerate(["b", "a", "b", "c", "a"])]
        assert rank_docs(hits) == ["b", "a", "c"]


class TestAnswerMetrics:
    def test_token_f1(self):
        assert token_f1("دماوند", "دماوند") == 1.0
        assert token_f1("البرز", "دماوند") == 0.0
        assert token_f1("", "دماوند") == 0.0
        # one of two content tokens matches on each side -> P = R = 0.5
        assert token_f1("قله دماوند", "کوه دماوند") == pytest.approx(0.5)

    def test_token_recall_does_not_punish_verbose_answers(self):
        verbose, ref = "تخت جمشید در نزدیکی شیراز قرار دارد", "شیراز"
        assert token_recall(verbose, ref) == 1.0
        assert token_f1(verbose, ref) < 0.5
        assert token_recall("", ref) == 0.0 and token_recall(verbose, "") == 0.0

    def test_token_f1_ignores_stopwords_and_orthographic_variants(self):
        assert token_f1("درياچه خزر است", "دریاچه‌ی خزر") == pytest.approx(0.8)  # P=1, R=2/3: normalization makes درياچه == دریاچه
        assert token_f1("كتاب", "کتاب") == 1.0

    def test_groundedness(self):
        ctx = ["قله دماوند در البرز است"]
        assert lexical_groundedness("دماوند در البرز", ctx) == 1.0
        assert lexical_groundedness("دماوند در هیمالیا", ctx) == pytest.approx(0.5)
        assert lexical_groundedness("", ctx) == 0.0
        assert lexical_groundedness("چیزی", []) == 0.0


def _chunks(*texts):
    return [Chunk(f"d{i}", 0, t) for i, t in enumerate(texts)]


class TestBM25:
    def test_ranks_matching_chunk_first(self):
        r = BM25Retriever()
        r.index(_chunks("دماوند بلندترین قله ایران است", "سعدی نویسنده گلستان است", "نوروز جشن بهار است"))
        hits = r.search("نویسنده گلستان کیست", 3)
        assert hits[0].chunk.doc_id == "d1"
        assert hits[0].score > 0

    def test_no_overlap_returns_nothing(self):
        r = BM25Retriever()
        r.index(_chunks("الف ب پ"))
        assert r.search("zzz", 5) == []
        assert r.search("", 5) == []

    def test_normalization_matches_arabic_keyboard_query(self):
        docs = _chunks("کتاب درباره دریاچه", "چیز دیگر")
        with_norm, without = BM25Retriever(), BM25Retriever(name="raw")
        without.tokenizer = type(without.tokenizer)(normalize=False)
        for r in (with_norm, without):
            r.index(docs)
        assert with_norm.search("درياچه", 1)[0].chunk.doc_id == "d0"
        assert without.search("درياچه", 1) == []

    def test_reindex_resets_state_and_is_deterministic(self):
        r = BM25Retriever()
        r.index(_chunks("الف ب", "ب ج"))
        first = [h.chunk.id for h in r.search("ب", 5)]
        r.index(_chunks("د ه"))
        assert r.search("ب", 5) == []
        r.index(_chunks("الف ب", "ب ج"))
        assert [h.chunk.id for h in r.search("ب", 5)] == first

    def test_char_ngram_variant_handles_morphology(self):
        docs = _chunks("کتابخانه ملی", "غذا و نوشیدنی")
        r = BM25Retriever(build_retriever("bm25-char3").tokenizer)
        r.index(docs)
        assert r.search("کتابخانه‌ها", 1)[0].chunk.doc_id == "d0"


class TestEmbeddingAndFusion:
    @staticmethod
    def toy_embed(texts):
        # 2-d "embedding": counts of two marker words
        return [[t.count("خزر") + 0.1, t.count("البرز") + 0.1] for t in texts]

    def test_embedding_retriever_cosine(self):
        r = EmbeddingRetriever(self.toy_embed, name="toy", batch_size=2)
        r.index(_chunks("خزر خزر", "البرز البرز", "خزر البرز", "هیچ"))
        assert r.search("خزر", 1)[0].chunk.doc_id == "d0"
        assert r.search("البرز", 1)[0].chunk.doc_id == "d1"
        assert len(r.search("خزر", 10)) == 4
        assert r.name == "toy"

    def test_embed_function_contract_is_checked(self):
        r = EmbeddingRetriever(lambda texts: [[1.0]])
        with pytest.raises(ValueError, match="returned 1 vectors for 2 texts"):
            r.index(_chunks("a", "b"))

    def test_rrf_prefers_documents_ranked_well_by_both(self):
        class Fixed:
            def __init__(self, name, order):
                self.name, self.order, self.chunks = name, order, {}

            def index(self, chunks):
                self.chunks = {c.doc_id: c for c in chunks}

            def search(self, query, k):
                return [Hit(self.chunks[d], 1.0) for d in self.order][:k]

        a = Fixed("a", ["d0", "d1", "d2"])
        b = Fixed("b", ["d1", "d0", "d2"])
        c = Fixed("c", ["d1", "d2", "d0"])
        fused = RRFRetriever([a, b, c])
        fused.index(_chunks("x", "y", "z"))
        assert [h.chunk.doc_id for h in fused.search("q", 3)] == ["d1", "d0", "d2"]
        assert fused.name == "rrf(a+b+c)"
        with pytest.raises(ValueError):
            RRFRetriever([])


class TestBuildRetriever:
    def test_builtin_specs(self):
        assert build_retriever("bm25").name == "bm25-word"
        assert build_retriever("bm25-raw").name == "bm25-word-raw"
        assert build_retriever("bm25-char4").name == "bm25-char4"

    def test_errors(self):
        for bad in ("nope", "bm25-charx", "no_such_module_xyz:make"):
            with pytest.raises(ValueError):
                build_retriever(bad)

    def test_custom_factory_from_module(self, tmp_path, monkeypatch):
        (tmp_path / "my_ret.py").write_text(
            "from ako_rageval import BM25Retriever\n"
            "def make():\n    r = BM25Retriever()\n    r.name = 'mine'\n    return r\n"
            "def broken():\n    return object()\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        assert build_retriever("my_ret:make").name == "mine"
        with pytest.raises(ValueError, match="must return an object"):
            build_retriever("my_ret:broken")
