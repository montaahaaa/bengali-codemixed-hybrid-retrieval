import csv, json, re, collections as C
import bm25s
SRC = "../dataset/benchmark_queries_10000_diverse - benchmark_queries_10000_diverse.csv.csv"
rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig")))
first = {}; recs = []
for r in rows:
    key = re.sub(r"\s+", " ", r["query"]).strip().lower()
    dup = first.get(key); first.setdefault(key, r["query_id"])
    recs.append({"_id": r["query_id"], "text": r["query"].strip(), "metadata": {
        "domain": r["domain"], "script": r["language_style"], "query_type": r["query_type"], "difficulty": r["difficulty"],
        "label_status": r["label_status"], "duplicate_of": dup if dup and dup != r["query_id"] else ""}})
with open("data/queries.jsonl", "w", encoding="utf-8") as f:
    for x in recs: f.write(json.dumps(x, ensure_ascii=False) + "\n")
uniq = [x for x in recs if not x["metadata"]["duplicate_of"]]
print("rows", len(recs), "unique", len(uniq))
tok = lambda t: [w for w in re.split(r"[\s\.,;:!?()\[\]\"'“”‘’।|/\\\-–—]+", t.lower()) if w]
ids = []; texts = []
for l in open("data/corpus.jsonl", encoding="utf-8"):
    d = json.loads(l); ids.append(d["doc_id"]); texts.append(d["text"])
ret = bm25s.BM25(); ret.index([tok(t) for t in texts], show_progress=False)
stat = C.defaultdict(list)
with open("data/candidates_bm25.jsonl", "w", encoding="utf-8") as f:
    for x in uniq:
        t = tok(x["text"])
        if not t: continue
        docs, sc = ret.retrieve([t], k=20, show_progress=False)
        f.write(json.dumps({"qid": x["_id"], "cands": [[ids[i], round(float(s), 3)] for i, s in zip(docs[0], sc[0])]}) + "\n")
        stat[(x["metadata"]["script"])].append(float(sc[0][0]))
import numpy as np
for k, v in stat.items(): print(k, len(v), "median top BM25 score", round(float(np.median(v)), 1), "| zero-hit", sum(1 for s in v if s == 0))
