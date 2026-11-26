from chunking import split_sentences, chunk_sentences, chunk_article, wc, MAX_WORDS


def test_bengali_split_on_danda_and_marks():
    t = "আমি ভাত খাই। তুমি কি যাবে? হ্যাঁ! ঠিক আছে।"
    assert split_sentences(t, "bn") == ["আমি ভাত খাই।", "তুমি কি যাবে?", "হ্যাঁ!", "ঠিক আছে।"]


def test_english_split_and_abbreviations():
    t = "Dr. Rahman went to St. Mary's school. He met Mr. Khan! Did it work? Yes."
    assert split_sentences(t, "en") == ["Dr. Rahman went to St. Mary's school.", "He met Mr. Khan!", "Did it work?", "Yes."]


def test_english_initial_not_a_boundary():
    assert split_sentences("J. K. Rowling wrote it. Fine.", "en") == ["J. K. Rowling wrote it.", "Fine."]


def test_newlines_are_boundaries():
    assert split_sentences("Heading\nBody text here.", "en") == ["Heading", "Body text here."]


def test_passages_never_exceed_max_unless_single_sentence():
    sents = ["word " * 40 + "end." for _ in range(10)]
    ps = chunk_sentences([s.strip() for s in sents])
    assert all(wc(p) <= MAX_WORDS for p in ps)
    assert sum(wc(p) for p in ps) == sum(wc(s) for s in sents)


def test_sentence_never_split_and_long_sentence_own_passage():
    long = " ".join(["w"] * 400) + "."
    ps = chunk_sentences(["Short one.", long, "Short two."])
    assert ps == ["Short one.", long, "Short two."]


def test_short_article_is_one_passage():
    assert chunk_article("Only one sentence here.", "en") == ["Only one sentence here."]


def test_empty_text_gives_no_passages():
    assert chunk_article("  \n \n", "bn") == []


def test_deterministic():
    t = "আমি ভাত খাই। " * 80
    assert chunk_article(t, "bn") == chunk_article(t, "bn")
