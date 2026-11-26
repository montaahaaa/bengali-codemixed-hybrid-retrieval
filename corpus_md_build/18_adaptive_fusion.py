"""Adaptive fusion v1: dense weight w chosen per query bucket (Bangla / mixed / Romanized, from the Latin-letter share of the query).
w is selected on a random 2,500-query TRAIN sample (nDCG@10 grid 0..1 step 0.1); dev and test are scored once with the chosen weights.
BM25 lane uses the skeleton normaliser (K=3). Baselines: fixed w=0.7 (normalised), dense only, raw hybrid 0.7."""
src = open("16_normaliser.py", encoding="utf-8").read().split("qrels = {")[0]
exec(src)
import csv as _csv, random
qrels = {r["query-id"]: idx[r["corpus-id"]] for r in _csv.DictReader(open("data/qrels.csv"))}
allq = [json.loads(l) for l in open("data/queries.jsonl", encoding="utf-8")]
bn = lambda s: len(re.findall(r"[ঀ-৿]", s)); la = lambda s: len(re.findall(r"[A-Za-z]", s))
def bucket(t):
    b, l = bn(t), la(t); r = l / (b + l) if b + l else 0
    return "bangla" if r < 0.05 else "romanized" if b == 0 else "mixed"
def mm(x): return (x - x.min()) / (x.max() - x.min() + 1e-9)
def bms(t):
    docs, sc = ret.retrieve([t], k=1000, show_progress=False); b = np.zeros(len(ids), dtype="float32"); b[docs[0]] = sc[0]; return b
def prep(qlist):
    out = []
    for s in range(0, len(qlist), 256):
        ch = qlist[s:s + 256]
        Q = model.encode(["query: " + q["text"] for q in ch], batch_size=64, normalize_embeddings=True, convert_to_tensor=True).cpu().float()
        D = (Q @ P.T).numpy()
        for qi, q in enumerate(ch):
            d = D[qi]; b3 = bms(expand(q["text"], 3)); b0 = bms(tok(q["text"]))
            c = np.array(sorted(set(np.argsort(-b3)[:1000]) | set(np.argsort(-d)[:1000]) | set(np.argsort(-b0)[:1000])))
            g = qrels[q["_id"]]; gi = int(np.where(c == g)[0][0]) if g in set(c.tolist()) else -1
            out.append(dict(id=q["_id"], bk=bucket(q["text"]), split=q["metadata"]["split"], dm=mm(d[c]), bn=mm(b3[c]), br=mm(b0[c]), dense=d[c], gi=gi))
        print(s + len(ch), "/", len(qlist), flush=True)
    return out
def ndcg(sc, gi):
    if gi < 0: return 0.0
    r = int((sc > sc[gi]).sum()); return 1 / math.log2(r + 2) if r < 10 else 0.0
W = [round(w, 1) for w in np.arange(0, 1.01, 0.1)]
random.seed(42)
train = [q for q in allq if q["metadata"]["split"] == "train" and tok(q["text"])]; random.shuffle(train); train = train[:2500]
T = prep(train)
best = {}; curve = {}
for bk in ("bangla", "mixed", "romanized"):
    qs = [x for x in T if x["bk"] == bk]
    curve[bk] = {w: float(np.mean([ndcg(w * x["dm"] + (1 - w) * x["bn"], x["gi"]) for x in qs])) for w in W}
    best[bk] = max(W, key=lambda w: curve[bk][w])
    print(f"train {bk:10s} n={len(qs):4d}  nDCG@10 by dense weight: " + " ".join(f"{w}:{curve[bk][w]:.3f}" for w in W), "-> chosen", best[bk])
E = prep([q for q in allq if q["metadata"]["split"] in ("dev", "test") and tok(q["text"])])
def run(x, name):
    if name == "adaptive": return ndcg(best[x["bk"]] * x["dm"] + (1 - best[x["bk"]]) * x["bn"], x["gi"])
    if name == "fixed0.7_norm": return ndcg(0.7 * x["dm"] + 0.3 * x["bn"], x["gi"])
    if name == "fixed0.7_raw": return ndcg(0.7 * x["dm"] + 0.3 * x["br"], x["gi"])
    if name == "dense": return ndcg(x["dense"], x["gi"])
SYS = ["dense", "fixed0.7_raw", "fixed0.7_norm", "adaptive"]; per = {}
print("\nchosen weights (dense):", best)
for sp in ("test", "dev"):
    print(f"\n{sp.upper()} nDCG@10"); print(f"{'bucket':11s}{'n':>5s}" + "".join(f"{s:>16s}" for s in SYS))
    for bk in ("bangla", "mixed", "romanized", "ALL"):
        qs = [x for x in E if x["split"] == sp and (bk == "ALL" or x["bk"] == bk)]
        row = f"{bk:11s}{len(qs):5d}"
        for s in SYS:
            v = [run(x, s) for x in qs]; row += f"{np.mean(v):.3f}".rjust(16)
            for x, vv in zip(qs, v): per[(x["id"], s)] = vv
        print(row)
rng = np.random.default_rng(42)
for sp in ("test", "dev"):
    for bk in ("bangla", "mixed", "romanized", "ALL"):
        ids_ = [x["id"] for x in E if x["split"] == sp and (bk == "ALL" or x["bk"] == bk)]
        for b_ in ("fixed0.7_norm", "dense"):
            d = np.array([per[(i, "adaptive")] - per[(i, b_)] for i in ids_]); bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
            print(f"paired {sp} {bk:9s} adaptive - {b_:13s}: {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] n={len(d)}")
json.dump({"chosen_weights": best, "train_curve": curve}, open("data/adaptive_fusion_results.json", "w"), indent=1)
