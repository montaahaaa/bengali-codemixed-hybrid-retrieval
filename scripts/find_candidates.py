"""Search aid for phase 1.5: show the top BM25 passages for a question.
Not used to score anything. The judgement of whether an answer exists is yours.
Usage: python scripts/find_candidates.py "your question" [--top 10]
The BM25 index (about 1.1M passages) is built on first use and cached in models/."""
import argparse, json, pickle, re
from pathlib import Path
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "models/bm25_pilot.pkl"
TOKEN = re.compile(r"[\wঀ-৿]+", re.U)  # keep Bengali vowel signs/virama inside words


def tok(s):
    return TOKEN.findall(s.lower())


def load():
    if CACHE.exists():
        return pickle.load(open(CACHE, "rb"))
    rows = [json.loads(l) for l in open(ROOT / "data/corpus/corpus.jsonl", encoding="utf-8")]
    bm = BM25Okapi([tok(r["title"] + " " + r["text"]) for r in rows])
    ids = [(r["passage_id"], r["title"], r["text"]) for r in rows]
    pickle.dump((bm, ids), open(CACHE, "wb"))
    return bm, ids


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()
    bm, ids = load()
    scores = bm.get_scores(tok(a.question))
    for i in scores.argsort()[::-1][: a.top]:
        pid, title, text = ids[i]
        print(f"{pid} | {title} | {scores[i]:.1f}\n  {text[:300]}\n")
