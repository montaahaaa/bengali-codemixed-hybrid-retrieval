"""Experiment 1 (pre-defence): BM25 over the full 1.1M-passage corpus, by query script (RQ1 pilot).
Usage: python scripts/exp1_bm25.py
Query sets (from data/benchmark_full, qrels in corpus only):
  synthetic: LLM queries written from a known passage (bn / latin / mixed). Labels exist by construction, so they are not biased towards BM25.
  real:      YouTube + Facebook queries with graded qrels (judged from BM25+reranker candidates, so BIASED towards BM25).
Writes experiments/exp1_bm25_results.json and appends rows to experiments/experiment_tracker.csv.
MIRACL queries are skipped (their passages are not in this corpus)."""
import csv, json, sys, time, array, datetime, collections as C
from pathlib import Path
import numpy as np, scipy.sparse as sp
sys.path.insert(0, str(Path(__file__).resolve().parent))
from find_candidates import load, tok
ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/benchmark_full/"
K, SEED = 100, 42
q = {r["qid"]: r for r in csv.DictReader(open(D / "queries.csv", encoding="utf-8-sig"))}
qr = C.defaultdict(dict)
for l in open(D / "qrels.tsv"):
    a, _, p, g = l.split(); qr[a][p] = int(g)
bm, ids = load(); pid = [x[0] for x in ids]; inc = set(pid)
vocab = {w: i for i, w in enumerate(bm.idf)}
k1, b, avgdl = bm.k1, bm.b, bm.avgdl
ptr, ind, dat = [0], array.array("i"), array.array("f")
for freqs, dl in zip(bm.doc_freqs, bm.doc_len):
    norm = k1 * (1 - b + b * dl / avgdl)
    for w, tf in freqs.items(): ind.append(vocab[w]); dat.append(bm.idf[w] * tf * (k1 + 1) / (tf + norm))
    ptr.append(len(ind))
X = sp.csr_matrix((np.frombuffer(dat, np.float32), np.frombuffer(ind, np.int32), np.array(ptr, np.int64)), shape=(len(pid), len(vocab))).tocsc()
print("index ready", flush=True)

sel = [a for a in qr if q[a]["source"] != "miracl_bn" and any(p in inc for p in qr[a])]
def metrics(qid):
    toks = [vocab[w] for w in dict.fromkeys(tok(q[qid]["query"])) if w in vocab]
    rel = {p: g for p, g in qr[qid].items() if p in inc and g >= 1}
    if not toks: return dict(r10=0, r100=0, mrr=0, ndcg=0, nohit=1)
    s = np.asarray(X[:, toks].sum(axis=1)).ravel()
    top = np.argpartition(-s, K)[:K]; top = top[np.argsort(-s[top])]
    ranked = [pid[i] for i in top if s[i] > 0]
    nrel = len(rel)
    hit = [i for i, p in enumerate(ranked) if p in rel]
    dcg = sum(rel[p] / np.log2(i + 2) for i, p in enumerate(ranked[:10]) if p in rel)
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(sorted(rel.values(), reverse=True)[:10]))
    return dict(r10=len([i for i in hit if i < 10]) / nrel, r100=len(hit) / nrel,
                mrr=(1 / (hit[0] + 1) if hit and hit[0] < 10 else 0), ndcg=dcg / idcg if idcg else 0, nohit=int(not ranked))
t0 = time.time(); res = {}
for k, a in enumerate(sel, 1):
    res[a] = metrics(a)
    if k % 100 == 0: print(k, len(sel), f"{time.time()-t0:.0f}s", flush=True)
rng = np.random.default_rng(SEED)
def agg(qs, m):
    v = np.array([res[a][m] for a in qs]); bs = [rng.choice(v, len(v)).mean() for _ in range(1000)]
    return round(float(v.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)
out = {}
for name, f in (("synthetic", lambda a: q[a]["source"].startswith("llm_")), ("real", lambda a: not q[a]["source"].startswith("llm_"))):
    for scr in ("bn", "mixed", "latin", "all"):
        qs = [a for a in sel if f(a) and (scr == "all" or q[a]["script"] == scr)]
        if not qs: continue
        out[f"{name}/{scr}"] = dict(n=len(qs), no_match=sum(res[a]["nohit"] for a in qs), **{m: agg(qs, m) for m in ("r10", "r100", "mrr", "ndcg")})
json.dump(out, open(ROOT / "experiments/exp1_bm25_results.json", "w"), indent=1)
with open(ROOT / "experiments/experiment_tracker.csv", "a", newline="") as f:
    w = csv.writer(f)
    for k, v in out.items():
        for m in ("r10", "r100", "mrr", "ndcg"): w.writerow([f"exp1_bm25/{k}", datetime.date.today().isoformat(), f"BM25 full corpus, n={v['n']}", m, v[m][0]])
for k, v in out.items(): print(k, v)
