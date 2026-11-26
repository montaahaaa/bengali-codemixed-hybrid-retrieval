"""Skeleton normaliser for Romanized Bangla (from scripts/exp2_normaliser.py) on the generated benchmark, dev+test.
Lookup built from the corpus only (no queries, no labels): every Bangla word gets a consonant skeleton; a Latin query token is
expanded with the K most frequent Bangla words that share its skeleton (original token kept). K=3 fixed in advance; K=1,5 sensitivity.
Systems: BM25, BM25+norm (K), dense E5 (raw query), hybrid = convex 0.7 / RRF of dense and BM25 (raw or normalised)."""
import csv, json, math, re, collections as C
import bm25s, numpy as np, torch
from sentence_transformers import SentenceTransformer
tok = lambda t: [w for w in re.split(r"[\s\.,;:!?()\[\]\"'“”‘’।|/\\\-–—]+", t.lower()) if w]
BN = {}
for chars, s in {"কখ": "k", "গঘ": "g", "চছ": "c", "জঝয": "j", "টঠতথৎ": "t", "ডঢদধ": "d", "ড়ঢ়": "r", "ঞণনঙং": "n",
                 "পফ": "p", "বভ": "b", "ম": "m", "য়": "y", "র": "r", "ল": "l", "শষস": "s", "হ": "h"}.items():
    for ch in chars: BN[ch] = s
def skel_bn(w):
    w = w.replace("ড়", "ড়").replace("ঢ়", "ঢ়").replace("য়", "য়"); out = []
    for i, ch in enumerate(w):
        if ch in BN and not (ch == "য" and i > 0 and w[i-1] == "্"): out.append(BN[ch])
    return re.sub(r"(.)\1+", r"\1", "".join(out))
def skel_lat(w):
    w = w.lower()
    for a, b in (("sh", "s"), ("ng", "n"), ("kh", "k"), ("gh", "g"), ("ph", "p"), ("bh", "b"), ("th", "t"), ("dh", "d"), ("jh", "j"), ("ch", "c")): w = w.replace(a, b)
    m = {"v": "b", "w": "b", "z": "j", "q": "k", "f": "p", "x": "k"}
    return re.sub(r"(.)\1+", r"\1", "".join(m.get(ch, ch) for ch in w if ch in "bcdfghjklmnpqrstvwxyz"))
ids = []; texts = []
for l in open("data/corpus.jsonl", encoding="utf-8"):
    r = json.loads(l); ids.append(r["doc_id"]); texts.append(r["text"])
idx = {d: i for i, d in enumerate(ids)}; toks = [tok(t) for t in texts]
freq = C.Counter(w for t in toks for w in t)
look = C.defaultdict(list)
for w, c in freq.items():
    if re.search(r"[ঀ-৿]", w) and not re.search(r"[A-Za-z0-9]", w) and len(w) >= 2:
        s = skel_bn(w)
        if len(s) >= 2: look[s].append((c, w))
look = {s: [w for _, w in sorted(v, reverse=True)[:5]] for s, v in look.items()}
print("lookup skeletons", len(look), flush=True)
def expand(text, K):
    out = []
    for w in dict.fromkeys(tok(text)):
        out.append(w)
        if K and re.fullmatch(r"[a-z]+", w) and len(w) >= 3: out += look.get(skel_lat(w), [])[:K]
    return list(dict.fromkeys(out))
ret = bm25s.BM25(); ret.index(toks, show_progress=False)
P = torch.from_numpy(np.load("data/emb_e5base_passages.npy").astype("float32"))
model = SentenceTransformer("intfloat/multilingual-e5-base", device="mps" if torch.backends.mps.is_available() else "cpu")
qrels = {r["query-id"]: idx[r["corpus-id"]] for r in csv.DictReader(open("data/v2/qrels.csv"))}
qs = [json.loads(l) for l in open("data/v2/queries.jsonl", encoding="utf-8")]
qs = [q for q in qs if q["metadata"]["split"] in ("dev", "test") and tok(q["text"])]
SYS = ["bm25", "bm25_norm_K1", "bm25_norm_K3", "bm25_norm_K5", "dense_e5", "convex0.7_raw", "convex0.7_norm", "rrf_raw", "rrf_norm"]
res = C.defaultdict(lambda: C.defaultdict(list))
def mm(x): return (x - x.min()) / (x.max() - x.min() + 1e-9)
def met(order, gold):
    pos = np.where(np.asarray(order[:100]) == gold)[0]; r = int(pos[0]) if len(pos) else None
    return {"R@10": float(r is not None and r < 10), "nDCG@10": 1 / math.log2(r + 2) if r is not None and r < 10 else 0.0, "MRR": 1 / (r + 1) if r is not None else 0.0}
def bm_scores(t):
    docs, sc = ret.retrieve([t], k=1000, show_progress=False); b = np.zeros(len(ids), dtype="float32"); b[docs[0]] = sc[0]; return b
def fuse(b, d):
    ob = np.argsort(-b)[:1000]; od = np.argsort(-d)[:1000]
    rb = {int(i): r for r, i in enumerate(ob)}; rd = {int(i): r for r, i in enumerate(od)}
    f = {i: 1 / (60 + rb.get(i, 1000)) + 1 / (60 + rd.get(i, 1000)) for i in set(rb) | set(rd)}
    cand = np.array(sorted(set(rb) | set(rd))); cv = cand[np.argsort(-(0.7 * mm(d[cand]) + 0.3 * mm(b[cand])))]
    return cv, sorted(f, key=f.get, reverse=True)
per_q = {}
for s in range(0, len(qs), 256):
    chunk = qs[s:s + 256]
    Q = model.encode(["query: " + q["text"] for q in chunk], batch_size=64, normalize_embeddings=True, convert_to_tensor=True).cpu().float()
    D = (Q @ P.T).numpy()
    for qi, q in enumerate(chunk):
        gold = qrels[q["_id"]]; d = D[qi]; m = q["metadata"]
        b0 = bm_scores(tok(q["text"])); b1 = bm_scores(expand(q["text"], 1)); b3 = bm_scores(expand(q["text"], 3)); b5 = bm_scores(expand(q["text"], 5))
        cv_r, rrf_r = fuse(b0, d); cv_n, rrf_n = fuse(b3, d)
        orders = {"bm25": np.argsort(-b0)[:100], "bm25_norm_K1": np.argsort(-b1)[:100], "bm25_norm_K3": np.argsort(-b3)[:100], "bm25_norm_K5": np.argsort(-b5)[:100],
                  "dense_e5": np.argsort(-d)[:100], "convex0.7_raw": cv_r, "convex0.7_norm": cv_n, "rrf_raw": rrf_r, "rrf_norm": rrf_n}
        for sname, o in orders.items():
            v = met(list(o), gold); per_q[(q["_id"], sname)] = v["nDCG@10"]
            for key in ((m["split"], m["script"]), (m["split"], "ALL")):
                for k2, x in v.items(): res[(key, sname)][k2].append(x)
    print(s + len(chunk), "/", len(qs), flush=True)
out = {}
for split in ("test", "dev"):
    print(f"\n{split.upper()} nDCG@10 / R@10")
    print(f"{'group':11s}{'n':>5s}" + "".join(f"{x:>16s}" for x in SYS))
    for g in ("bangla", "mixed", "romanized", "ALL"):
        n = len(res[((split, g), "bm25")]["R@10"]); row = f"{g:11s}{n:5d}"
        for sname in SYS:
            r = res[((split, g), sname)]; out[f"{split}|{g}|{sname}"] = {k: float(np.mean(v)) for k, v in r.items()} | {"n": n}
            row += f"{np.mean(r['nDCG@10']):.3f}/{np.mean(r['R@10']):.3f}".rjust(16)
        print(row)
rng = np.random.default_rng(42)
for g in ("romanized", "mixed", "bangla"):
    ids_g = [q["_id"] for q in qs if q["metadata"]["script"] == g]
    for a, b_ in (("bm25_norm_K3", "bm25"), ("convex0.7_norm", "convex0.7_raw"), ("convex0.7_norm", "dense_e5")):
        d = np.array([per_q[(i, a)] - per_q[(i, b_)] for i in ids_g]); bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
        print(f"paired nDCG@10 {g:9s} {a} - {b_}: {d.mean():+.3f}  95% CI [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]  (dev+test, n={len(d)})")
json.dump(out, open("data/v2/normaliser_results.json", "w"), indent=1)
