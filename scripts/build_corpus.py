"""Build passages for one language: python build_corpus.py bn|en --snapshot 20231101
Writes data/corpus/corpus_<lang>.jsonl. Combine with --merge to make corpus.jsonl."""
import argparse, glob, json
from pathlib import Path
import pandas as pd
from chunking import chunk_article

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/wikipedia"
OUT = ROOT / "data/corpus"


def articles(lang, snapshot):
    if lang == "bn":
        for f in sorted(glob.glob(str(RAW / f"{snapshot}.bn/*.parquet"))):
            for r in pd.read_parquet(f).itertuples():
                yield r.id, r.title, r.text
    else:
        for line in open(RAW / "en_filtered.jsonl", encoding="utf-8"):
            d = json.loads(line)
            yield d["id"], d["title"], d["text"]


def build(lang, snapshot):
    n_art = n_pass = 0
    with open(OUT / f"corpus_{lang}.jsonl", "w", encoding="utf-8") as fo:
        for aid, title, text in articles(lang, snapshot):
            n_art += 1
            for i, p in enumerate(chunk_article(text, lang)):
                fo.write(json.dumps({"passage_id": f"{lang}-{aid}-{i}", "lang": lang, "title": title,
                                     "text": p, "source_article_id": aid, "snapshot": snapshot},
                                    ensure_ascii=False) + "\n")
                n_pass += 1
    print(lang, "articles", n_art, "passages", n_pass)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("lang", nargs="?", choices=["bn", "en"])
    ap.add_argument("--snapshot", default="20231101")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        with open(OUT / "corpus.jsonl", "w", encoding="utf-8") as fo:
            for l in ("bn", "en"):
                fo.write(open(OUT / f"corpus_{l}.jsonl", encoding="utf-8").read())
    else:
        build(a.lang, a.snapshot)
