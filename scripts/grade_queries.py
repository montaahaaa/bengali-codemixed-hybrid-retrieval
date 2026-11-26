"""Search the pilot corpus for every mined query and give an automatic 0/1/2 grade (phase 2.3 helper).
Usage: python scripts/grade_queries.py
Reads data/processed/youtube_queries_5000.csv and the Facebook rows of data/processed/mining_pilot.jsonl,
runs BM25 over the 1.1M-passage corpus (same index/tokenizer as find_candidates.py, but via a sparse matrix so
5.7k queries take minutes) and writes data/processed/graded_queries.csv (git-ignored, private text).
The grade is a LEXICAL PROXY, not a judgement of whether an answer exists. A human checks every row:
  2 = query has >=3 informative words and one top-5 passage contains >=60% of them (at least 3)
  1 = query has >=2 informative words and a top-5 passage contains >=40% of them (at least 2): topical only
  0 = otherwise (nothing related, or the query is Romanized Bangla that BM25 cannot match)
Informative word = idf >= 4.5 in the corpus (in under about 1% of passages), so ordinary words like কি, হবে, ভাই do not count."""
import csv, json, pickle, sys, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from find_candidates import load, tok

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/processed/graded_queries.csv"
IDF_MIN, TOPK, NCAND = 4.5, 5, 10

t0 = time.time()
bm, ids = load()
print(f"index loaded {time.time()-t0:.0f}s, {len(ids)} passages", flush=True)

vocab = {w: i for i, w in enumerate(bm.idf)}
n = len(ids)
k1, b, avgdl = bm.k1, bm.b, bm.avgdl
indptr, indices, data = [0], __import__("array").array("i"), __import__("array").array("f")
for d, (freqs, dl) in enumerate(zip(bm.doc_freqs, bm.doc_len)):
    norm = k1 * (1 - b + b * dl / avgdl)
    for w, tf in freqs.items():
        indices.append(vocab[w])
        data.append(bm.idf[w] * tf * (k1 + 1) / (tf + norm))
    indptr.append(len(indices))
X = sp.csr_matrix((np.frombuffer(data, dtype=np.float32), np.frombuffer(indices, dtype=np.int32),
                   np.array(indptr, dtype=np.int64)), shape=(n, len(vocab))).tocsc()
print(f"sparse matrix built {time.time()-t0:.0f}s, nnz={X.nnz}", flush=True)

REST = "--rest" in sys.argv
_sel = ROOT / "data/processed/dataset_10000_ids.json"
SEL = set(json.load(open(_sel))) if REST and _sel.exists() else None  # grade only likely_need items not graded in the first run
rows = []
if not REST:
    with open(ROOT / "data/processed/youtube_queries_5000.csv", encoding="utf-8-sig") as f:
        rows += list(csv.DictReader(f))
done = {r["id"] for r in csv.DictReader(open(ROOT / "data/processed/graded_queries.csv", encoding="utf-8-sig"))} if REST else set()
for l in open(ROOT / "data/processed/mining_pilot.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if REST and r["likely_need"] and r["id"] not in done and (not SEL or r["id"] in SEL):
        rows.append(r)
    elif not REST and r["source"] == "facebook" and r["likely_need"]:
        rows.append(r)
if REST:
    OUT = ROOT / "data/processed/graded_queries_rest.csv"

cands, cur = {}, [None]


def grade(q):
    toks = list(dict.fromkeys(tok(q)))
    inf = [w for w in toks if w in vocab and bm.idf[w] >= IDF_MIN]
    qid = [vocab[w] for w in toks if w in vocab]
    if not qid:
        return 0, inf, 0, 0.0, []
    s = np.asarray(X[:, qid].sum(axis=1)).ravel()
    top = np.argpartition(-s, NCAND)[:NCAND]
    top = top[np.argsort(-s[top])]
    best_cov, best_m = 0.0, 0
    for i in top[:TOPK]:
        m = sum(1 for w in inf if w in bm.doc_freqs[i])
        c = m / len(inf) if inf else 0.0
        if (c, m) > (best_cov, best_m):
            best_cov, best_m = c, m
    g = 0
    if len(inf) >= 3 and best_cov >= 0.6 and best_m >= 3:
        g = 2
    elif len(inf) >= 2 and best_cov >= 0.4 and best_m >= 2:
        g = 1
    cands[cur[0]] = [[ids[i][0], ids[i][1], ids[i][2][:700]] for i in top]
    return g, inf, best_m, best_cov, [(ids[i], float(s[i])) for i in top[:3]]

cols = ["id", "source", "domain", "script", "text", "auto_grade", "informative_terms", "matched", "coverage",
        "p1_id", "p1_title", "p1_text", "p2_id", "p2_title", "p2_text", "p3_id", "p3_title", "p3_text",
        "manual_grade", "notes"]
with OUT.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(cols)
    for k, r in enumerate(rows, 1):
        cur[0] = r["id"]
        g, inf, m, c, top = grade(r["text"])
        flat = []
        for j in range(3):
            if j < len(top):
                (pid, title, text), _ = top[j]
                flat += [pid, title, text[:400]]
            else:
                flat += ["", "", ""]
        w.writerow([r["id"], r["source"], r.get("domain") or "", r["script"], r["text"], g, " ".join(inf), m,
                    f"{c:.2f}", *flat, "", ""])
        if k % 500 == 0:
            print(f"{k}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
json.dump(cands, open(ROOT / ("data/processed/bm25_candidates" + ("_rest" if REST else "") + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
print("done", len(rows))
