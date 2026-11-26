import csv, json, math, re, collections as C
import bm25s, numpy as np
tok = lambda t: [w for w in re.split(r"[\s\.,;:!?()\[\]\"'“”‘’।|/\\\-–—]+", t.lower()) if w]
ids, texts = [], []
for l in open("data/corpus.jsonl", encoding="utf-8"):
    r = json.loads(l); ids.append(r["doc_id"]); texts.append(r["text"])
ret = bm25s.BM25(); ret.index([tok(t) for t in texts], show_progress=False)
qs = [json.loads(l) for l in open("data/queries.jsonl", encoding="utf-8")]
qrels = C.defaultdict(dict)
for r in csv.DictReader(open("data/qrels.csv", encoding="utf-8")): qrels[r["query-id"]][r["corpus-id"]] = max(int(r["score"]), qrels[r["query-id"]].get(r["corpus-id"], 0))
def group(q):
    m = q["metadata"]
    if m["source"].startswith("draft_"): return "draft_" + m["source"][6:]
    if m["eval_ok"] == "yes": return "real_eval_ok_" + m["script"]
    return None
res = C.defaultdict(lambda: C.defaultdict(list)); runs = {}
for q in qs:
    g = group(q)
    if not g or q["_id"] not in qrels: continue
    toks = tok(q["text"])
    if not toks: continue
    docs, sc = ret.retrieve([toks], k=100, show_progress=False)
    top = [ids[i] for i in docs[0]]; rel = qrels[q["_id"]]; runs[q["_id"]] = top[:100]
    rank = {d: i for i, d in enumerate(top)}
    r10 = sum(1 for d in rel if rank.get(d, 99) < 10) / len(rel)
    dcg = sum(rel[d] / math.log2(rank[d] + 2) for d in rel if rank.get(d, 99) < 10)
    idcg = sum(g_ / math.log2(i + 2) for i, g_ in enumerate(sorted(rel.values(), reverse=True)[:10]))
    first = min([rank[d] for d in rel if d in rank], default=None)
    for key in (g, "ALL"): 
        res[key]["R@10"].append(r10); res[key]["nDCG@10"].append(dcg / idcg); res[key]["MRR@100"].append(1 / (first + 1) if first is not None else 0)
print(f"{'group':26s}{'n':>6s}{'R@10':>8s}{'nDCG@10':>9s}{'MRR':>8s}")
out = {}
for g in sorted(res):
    n = len(res[g]["R@10"]); m = {k: float(np.mean(v)) for k, v in res[g].items()}; out[g] = {"n": n, **m}
    print(f"{g:26s}{n:6d}{m['R@10']:8.3f}{m['nDCG@10']:9.3f}{m['MRR@100']:8.3f}")
json.dump(out, open("data/bm25_results.json", "w"), indent=1); json.dump(runs, open("data/bm25_run.json", "w"))
