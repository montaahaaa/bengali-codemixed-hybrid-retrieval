"""Extra analysis on the test split: nDCG@10 by domain and query length, lexical overlap of query and gold passage, rank headroom.
Systems: BM25, dense E5, adaptive fusion (normalised BM25, weights 0.8/0.9/0.7). Writes data/analysis_extra.json."""
src = open("16_normaliser.py", encoding="utf-8").read().split("qrels = {")[0]
exec(src)
import csv as _csv, json, math, collections as C
qrels = {r["query-id"]: idx[r["corpus-id"]] for r in _csv.DictReader(open("data/qrels.csv"))}
qs = [json.loads(l) for l in open("data/queries.jsonl", encoding="utf-8")]
qs = [q for q in qs if q["metadata"]["split"] == "test" and tok(q["text"])]
W = {"bangla": .8, "mixed": .9, "romanized": .7}
def mm(x): return (x - x.min()) / (x.max() - x.min() + 1e-9)
def bms(t):
    docs, sc = ret.retrieve([t], k=1000, show_progress=False); b = np.zeros(len(ids), dtype="float32"); b[docs[0]] = sc[0]; return b
def nd(sc, gold):
    r = int((sc > sc[gold]).sum() + (sc == sc[gold]).sum() - 1)   # ties count against the system
    return (1 / math.log2(r + 2) if r < 10 else 0.0), r
rows = []
for s in range(0, len(qs), 256):
    ch = qs[s:s + 256]
    Q = model.encode(["query: " + q["text"] for q in ch], batch_size=64, normalize_embeddings=True, convert_to_tensor=True).cpu().float()
    D = (Q @ P.T).numpy()
    for qi, q in enumerate(ch):
        g = qrels[q["_id"]]; d = D[qi]; b0 = bms(tok(q["text"])); b3 = bms(expand(q["text"], 3)); m = q["metadata"]
        c = np.array(sorted(set(np.argsort(-b3)[:1000]) | set(np.argsort(-d)[:1000]))); gi = np.where(c == g)[0]
        ad = 0.0; rk = None
        if len(gi):
            sc = W[m["script"]] * mm(d[c]) + (1 - W[m["script"]]) * mm(b3[c]); ad, rk = nd(sc, gi[0])
        qt = set(tok(q["text"])); gt = set(toks[g]); ov = len(qt & gt) / max(1, len(qt))
        rows.append(dict(script=m["script"], domain=m["domain"], nw=len(q["text"].split()), bm25=nd(b0, g)[0], dense=nd(d, g)[0], adaptive=ad, adaptive_rank=rk, overlap=ov))
import statistics as st
out = {}
def agg(rs): return {"n": len(rs), "bm25": round(st.mean(r["bm25"] for r in rs), 3), "dense": round(st.mean(r["dense"] for r in rs), 3), "adaptive": round(st.mean(r["adaptive"] for r in rs), 3)}
for sc in ("bangla", "mixed", "romanized"): out["script|" + sc] = agg([r for r in rows if r["script"] == sc])
dom = C.Counter(r["domain"] for r in rows)
for d_, n in dom.most_common():
    if n >= 120: out["domain|" + d_] = agg([r for r in rows if r["domain"] == d_])
for lab, lo, hi in (("<=7 words", 0, 7), ("8-11", 8, 11), (">=12", 12, 99)):
    out["length|" + lab] = agg([r for r in rows if lo <= r["nw"] <= hi])
for sc in ("bangla", "mixed", "romanized"):
    rs = [r for r in rows if r["script"] == sc]; out["overlap|" + sc] = round(st.mean(r["overlap"] for r in rs), 3)
    rk = [r["adaptive_rank"] for r in rs if r["adaptive_rank"] is not None]
    out["adaptive_gold_rank|" + sc] = {"in_top10": round(sum(x < 10 for x in rk) / len(rs), 3), "in_top100": round(sum(x < 100 for x in rk) / len(rs), 3), "in_candidates": round(len(rk) / len(rs), 3)}
json.dump(out, open("data/analysis_extra.json", "w"), indent=1)
for k, v in out.items(): print(k, v)
