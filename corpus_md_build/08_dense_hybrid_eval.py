import csv, json, math, re, collections as C
import bm25s, numpy as np, torch
from sentence_transformers import SentenceTransformer
tok = lambda t: [w for w in re.split(r"[\s\.,;:!?()\[\]\"'“”‘’।|/\\\-–—]+", t.lower()) if w]
ids, texts = [], []
for l in open("data/corpus.jsonl", encoding="utf-8"):
    r = json.loads(l); ids.append(r["doc_id"]); texts.append(r["text"])
ret = bm25s.BM25(); ret.index([tok(t) for t in texts], show_progress=False)
P = np.load("data/emb_e5base_passages.npy").astype("float32")
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer("intfloat/multilingual-e5-base", device=dev)
qs = [json.loads(l) for l in open("data/queries.jsonl", encoding="utf-8")]
qrels = C.defaultdict(dict)
for r in csv.DictReader(open("data/qrels.csv", encoding="utf-8")): qrels[r["query-id"]][r["corpus-id"]] = max(int(r["score"]), qrels[r["query-id"]].get(r["corpus-id"], 0))
def group(q):
    m = q["metadata"]
    if m["source"].startswith("draft_"): return "draft_" + m["source"][6:]
    return "real_eval_ok_" + m["script"] if m["eval_ok"] == "yes" else None
sel = [q for q in qs if group(q) and q["_id"] in qrels and tok(q["text"])]
Q = model.encode(["query: " + q["text"] for q in sel], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
D = Q @ P.T                                    # dense scores, all passages
def mm(x): lo, hi = x.min(), x.max(); return (x - lo) / (hi - lo + 1e-9)
systems = ["bm25", "dense_e5", "rrf", "convex_0.3", "convex_0.5", "convex_0.7"]
res = C.defaultdict(lambda: C.defaultdict(list))
for qi, q in enumerate(sel):
    docs, sc = ret.retrieve([tok(q["text"])], k=1000, show_progress=False)
    b = np.zeros(len(ids), dtype="float32"); b[docs[0]] = sc[0]          # BM25 top-1000 scores
    d = D[qi]
    lists = {"bm25": np.argsort(-b)[:100], "dense_e5": np.argsort(-d)[:100]}
    rb = {i: r for r, i in enumerate(np.argsort(-b)[:1000])}; rd = {i: r for r, i in enumerate(np.argsort(-d)[:1000])}
    fused = {i: 1 / (60 + rb.get(i, 1000)) + 1 / (60 + rd.get(i, 1000)) for i in set(rb) | set(rd)}
    lists["rrf"] = np.array(sorted(fused, key=fused.get, reverse=True)[:100])
    cand = np.array(sorted(set(rb) | set(rd)))
    bm, dm = mm(b[cand]), mm(d[cand])
    for w in (0.3, 0.5, 0.7):               # w = weight on dense
        lists[f"convex_{w}"] = cand[np.argsort(-(w * dm + (1 - w) * bm))[:100]]
    rel = qrels[q["_id"]]
    for s in systems:
        top = [ids[i] for i in lists[s]]; rank = {x: i for i, x in enumerate(top)}
        r10 = sum(1 for x in rel if rank.get(x, 99) < 10) / len(rel)
        dcg = sum(rel[x] / math.log2(rank[x] + 2) for x in rel if rank.get(x, 99) < 10)
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(rel.values(), reverse=True)[:10]))
        for key in (group(q), "ALL"): res[(key, s)]["R@10"].append(r10); res[(key, s)]["nDCG@10"].append(dcg / idcg)
groups = sorted({k[0] for k in res}); out = {}
print(f"{'group':24s}{'n':>6s}  " + "".join(f"{s:>12s}" for s in systems) + "   (nDCG@10 / R@10)")
for g in groups:
    n = len(res[(g, 'bm25')]['R@10']); row = f"{g:24s}{n:6d}  "
    for s in systems:
        a = res[(g, s)]; out[f"{g}|{s}"] = {"n": n, "nDCG@10": float(np.mean(a['nDCG@10'])), "R@10": float(np.mean(a['R@10']))}
        row += f"{np.mean(a['nDCG@10']):.2f}/{np.mean(a['R@10']):.2f}".rjust(12)
    print(row)
json.dump(out, open("data/dense_hybrid_results.json", "w"), indent=1)
