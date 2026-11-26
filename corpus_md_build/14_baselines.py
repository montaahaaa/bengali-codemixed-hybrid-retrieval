"""BM25 / dense (multilingual-E5-base) / hybrid (RRF, convex 0.3/0.5/0.7) on dev+test queries of the generated benchmark."""
import csv, json, math, re, collections as C
import bm25s, numpy as np, torch
from sentence_transformers import SentenceTransformer
tok = lambda t: [w for w in re.split(r"[\s\.,;:!?()\[\]\"'“”‘’।|/\\\-–—]+", t.lower()) if w]
ids = []; texts = []
for l in open("data/corpus.jsonl", encoding="utf-8"):
    r = json.loads(l); ids.append(r["doc_id"]); texts.append(r["text"])
idx = {d: i for i, d in enumerate(ids)}
ret = bm25s.BM25(); ret.index([tok(t) for t in texts], show_progress=False)
P = torch.from_numpy(np.load("data/emb_e5base_passages.npy").astype("float32"))
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer("intfloat/multilingual-e5-base", device=dev)
qrels = {r["query-id"]: idx[r["corpus-id"]] for r in csv.DictReader(open("data/qrels.csv"))}
qs = [json.loads(l) for l in open("data/queries.jsonl", encoding="utf-8")]
qs = [q for q in qs if q["metadata"]["split"] in ("dev", "test") and tok(q["text"])]
print("queries", len(qs), flush=True)
SYS = ["bm25", "dense_e5", "rrf", "convex_0.3", "convex_0.5", "convex_0.7"]
res = C.defaultdict(lambda: C.defaultdict(list))
def mm(x): return (x - x.min()) / (x.max() - x.min() + 1e-9)
def score(key_list, gold, group):
    rank = {int(i): r for r, i in enumerate(key_list[:100])}
    r = rank.get(gold, None)
    return [(1.0 if r is not None and r < 10 else 0.0), (1 / math.log2(r + 2) if r is not None and r < 10 else 0.0),
            (1 / (r + 1) if r is not None else 0.0), (1.0 if r is not None else 0.0)]
CH = 256
for s in range(0, len(qs), CH):
    chunk = qs[s:s + CH]
    Q = model.encode(["query: " + q["text"] for q in chunk], batch_size=64, normalize_embeddings=True, convert_to_tensor=True).cpu().float()
    D = (Q @ P.T).numpy()
    for qi, q in enumerate(chunk):
        gold = qrels[q["_id"]]
        docs, sc = ret.retrieve([tok(q["text"])], k=1000, show_progress=False)
        b = np.zeros(len(ids), dtype="float32"); b[docs[0]] = sc[0]; d = D[qi]
        ob = np.argsort(-b)[:1000]; od = np.argsort(-d)[:1000]
        rb = {int(i): r for r, i in enumerate(ob)}; rd = {int(i): r for r, i in enumerate(od)}
        fused = {i: 1 / (60 + rb.get(i, 1000)) + 1 / (60 + rd.get(i, 1000)) for i in set(rb) | set(rd)}
        cand = np.array(sorted(set(rb) | set(rd))); bm, dm = mm(b[cand]), mm(d[cand])
        lists = {"bm25": ob, "dense_e5": od, "rrf": sorted(fused, key=fused.get, reverse=True)}
        for w in (0.3, 0.5, 0.7): lists[f"convex_{w}"] = cand[np.argsort(-(w * dm + (1 - w) * bm))]
        m = q["metadata"]
        for sname in SYS:
            v = score(list(lists[sname]), gold, None)
            for key in ((m["split"], m["script"]), (m["split"], "ALL")):
                for name, x in zip(("R@10", "nDCG@10", "MRR@100", "R@100"), v): res[(key, sname)][name].append(x)
    print(s + len(chunk), "/", len(qs), flush=True)
out = {}
for split in ("test", "dev"):
    print(f"\n{split.upper()}  (nDCG@10 / R@10 / R@100)")
    print(f"{'group':12s}{'n':>6s}" + "".join(f"{x:>18s}" for x in SYS))
    for g in ("bangla", "mixed", "romanized", "ALL"):
        a = res[((split, g), "bm25")]; n = len(a["R@10"])
        row = f"{g:12s}{n:6d}"
        for sname in SYS:
            r = res[((split, g), sname)]; out[f"{split}|{g}|{sname}"] = {k: float(np.mean(v)) for k, v in r.items()} | {"n": n}
            row += f"{np.mean(r['nDCG@10']):.3f}/{np.mean(r['R@10']):.3f}/{np.mean(r['R@100']):.3f}".rjust(18)
        print(row)
json.dump(out, open("data/baseline_results.json", "w"), indent=1)
