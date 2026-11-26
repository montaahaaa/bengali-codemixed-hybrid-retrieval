"""Split article text into sentences and group them into passages of about
100-150 words. A sentence is never split. Words are whitespace-separated tokens."""
import re

MIN_WORDS = 100
MAX_WORDS = 150

# Abbreviations whose trailing full stop does not end an English sentence.
ABBREV = {"mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc", "no", "inc", "ltd",
          "co", "corp", "mt", "gen", "col", "lt", "capt", "sgt", "rev", "hon", "fig", "approx",
          "e.g", "i.e", "u.s", "u.k"}

_BN_SPLIT = re.compile(r"(?<=[।?!])\s+")
_EN_CAND = re.compile(r"(?<=[.?!])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")


def _is_abbrev_end(chunk):
    last = chunk.rstrip().rstrip("\"')]").rsplit(None, 1)[-1] if chunk.strip() else ""
    word = last.rstrip(".").lower()
    return last.endswith(".") and (word in ABBREV or (len(word) == 1 and word.isalpha()))


def split_sentences(text, lang):
    sentences = []
    for line in re.split(r"\n+", text):
        line = line.strip()
        if not line:
            continue
        if lang == "bn":
            sentences += [s.strip() for s in _BN_SPLIT.split(line) if s.strip()]
        else:
            start, parts = 0, []
            for m in _EN_CAND.finditer(line):
                piece = line[start:m.end()]
                if _is_abbrev_end(piece):
                    continue
                parts.append(piece.strip())
                start = m.end()
            parts.append(line[start:].strip())
            sentences += [p for p in parts if p]
    return sentences


def wc(s):
    return len(s.split())


def chunk_sentences(sentences):
    """Greedy grouping. A passage is closed when the next sentence would push it
    past MAX_WORDS. A sentence longer than MAX_WORDS becomes its own passage."""
    passages, cur, n = [], [], 0
    for s in sentences:
        w = wc(s)
        if cur and n + w > MAX_WORDS:
            passages.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        passages.append(" ".join(cur))
    return passages


def chunk_article(text, lang):
    return chunk_sentences(split_sentences(text, lang))
