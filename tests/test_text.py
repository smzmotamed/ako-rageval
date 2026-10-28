import pytest

from ako_rageval import Tokenizer, normalize, parse_chunker, split_sentences
from ako_rageval.chunking import FixedSizeChunker, ParagraphChunker, SentenceChunker, WholeDocChunker
from ako_rageval.tokenizer import char_ngrams


class TestNormalize:
    def test_arabic_letters_become_persian(self):
        # Arabic yeh/kaf (U+064A, U+0643) typed on an Arabic keyboard
        assert normalize("كتاب درياچه") == "کتاب دریاچه"

    def test_digits_to_ascii_by_default(self):
        assert normalize("۱۲۳ ٤٥٦ 789") == "123 456 789"

    def test_digits_to_persian(self):
        assert normalize("123 ٤٥٦", digits="persian") == "۱۲۳ ۴۵۶"

    def test_diacritics_and_tatweel_removed(self):
        assert normalize("مُحَمَّد") == "محمد"
        assert normalize("سلـــام") == "سلام"

    def test_zwnj_policies(self):
        text = "می‌روم"
        assert normalize(text) == "می روم"
        assert normalize(text, zwnj="keep") == "می‌روم"
        assert normalize(text, zwnj="remove") == "میروم"
        assert normalize("می روم") == normalize(text)

    def test_invalid_options_raise(self):
        with pytest.raises(ValueError):
            normalize("x", zwnj="bad")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            normalize("x", digits="bad")  # type: ignore[arg-type]

    def test_punctuation_bidi_marks_and_whitespace(self):
        assert normalize("«سلام»،‏  دنیا؟") == '"سلام", دنیا?'

    def test_alef_madda_folding_is_opt_in(self):
        assert normalize("آب") == "آب"
        assert normalize("آب", fold_alef_madda=True) == "اب"

    def test_presentation_forms_are_folded(self):
        assert normalize("ﻟﻤ") == normalize("لم")  # isolated/initial forms -> base letters

    def test_idempotent_and_empty(self):
        s = "كتابهاى  ۱۲۳ مُحَمَّد، «آزمون»؟"
        once = normalize(s)
        assert normalize(once) == once
        assert normalize("") == ""

    def test_latin_is_lowercased(self):
        assert normalize("Kafka و NATS") == "kafka و nats"


class TestTokenizer:
    def test_stopwords_removed_and_normalized(self):
        toks = Tokenizer()("بلندترین قله‌ی ایران کدام است؟")
        assert "کدام" not in toks and "است" not in toks
        assert "قله" in toks and "ایران" in toks

    def test_char_ngrams(self):
        assert char_ngrams("کتاب", 3) == ["#کت", "کتا", "تاب", "اب#"]
        assert char_ngrams("a", 3) == ["#a#"]

    def test_char_mode_and_names(self):
        t = Tokenizer(mode="char", ngram=3)
        assert t.name == "char3"
        assert t("کتاب") == ["#کت", "کتا", "تاب", "اب#"]
        assert Tokenizer(normalize=False).name == "word-raw"

    def test_raw_mode_does_not_unify_arabic_letters(self):
        assert Tokenizer(normalize=False)("كتاب") != Tokenizer()("كتاب")

    def test_invalid_config(self):
        with pytest.raises(ValueError):
            Tokenizer(mode="bogus")
        with pytest.raises(ValueError):
            Tokenizer(mode="char", ngram=1)


class TestChunking:
    def test_sentence_splitter_handles_persian_question_mark(self):
        text = "این چیست؟ یک آزمون است. آیا کار می‌کند! بله…  خوب"
        assert split_sentences(text) == ["این چیست؟", "یک آزمون است.", "آیا کار می‌کند!", "بله…", "خوب"]

    def test_decimals_are_not_split(self):
        assert split_sentences("عدد 3.14 است. تمام.") == ["عدد 3.14 است.", "تمام."]

    def test_sentence_chunker_groups_and_overlaps(self):
        text = "الف. ب. ج. د. ه."
        assert [c.text for c in SentenceChunker(2).split("d", text)] == ["الف. ب.", "ج. د.", "ه."]
        assert [c.text for c in SentenceChunker(3, overlap=1).split("d", text)] == ["الف. ب. ج.", "ج. د. ه."]

    def test_fixed_chunker_windows(self):
        words = " ".join(f"w{i}" for i in range(10))
        chunks = FixedSizeChunker(size=4, overlap=2).split("d", words)
        assert [c.text for c in chunks] == ["w0 w1 w2 w3", "w2 w3 w4 w5", "w4 w5 w6 w7", "w6 w7 w8 w9"]
        assert [c.index for c in chunks] == [0, 1, 2, 3]
        assert chunks[1].id == "d#1"

    def test_fixed_chunker_short_text_single_chunk(self):
        assert [c.text for c in FixedSizeChunker(size=100, overlap=10).split("d", "a b c")] == ["a b c"]

    def test_paragraph_chunker_merges_short_paragraphs(self):
        text = "یک\n\nدو سه چهار پنج\n\nشش"
        assert [c.text for c in ParagraphChunker().split("d", text)] == ["یک", "دو سه چهار پنج", "شش"]
        assert [c.text for c in ParagraphChunker(min_words=3).split("d", text)] == ["یک دو سه چهار پنج شش"]

    def test_whole_doc_and_empty(self):
        assert [c.text for c in WholeDocChunker().split("d", "a\n b")] == ["a b"]
        for chunker in (WholeDocChunker(), FixedSizeChunker(), SentenceChunker(), ParagraphChunker()):
            assert chunker.split("d", "   \n ") == []

    def test_spec_parsing(self):
        assert parse_chunker("doc").name == "doc"
        assert parse_chunker("paragraph").name == "paragraph"
        assert parse_chunker("sentence").name == "sentence:3"
        assert parse_chunker("sentence:2:1").name == "sentence:2:1"
        assert parse_chunker("fixed:50:5").name == "fixed:50:5"
        for bad in ("bogus", "fixed:a", "fixed:10:10", "sentence:2:2", "doc:1", "fixed:1:2:3"):
            with pytest.raises(ValueError):
                parse_chunker(bad)
