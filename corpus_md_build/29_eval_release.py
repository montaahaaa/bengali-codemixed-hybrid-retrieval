"""Evaluate the released benchmark (submission/*.csv).

Follows corpus_md_build/14v2_baselines.py, 16v2_normaliser.py, 18v2_adaptive_fusion.py, 17_reviewed_rerun.py and
24_extra_analysis.py, with three changes made so that every table uses the same rules:
  1. Input is the released corpus (99,567 passages) and the released queries, not the 100,000-passage working files.
  2. A gold passage tied with other passages is ranked after all of them, for every system.
  3. Adaptive fusion uses the same candidate pool as the fixed-weight fusion, so w = 0.7 reproduces it exactly.

Usage (from corpus_md_build): python 29_eval_release.py .. data/release_eval/release_eval.json
"""
import csv, json, math, random, re, sys, collections as C
from pathlib import Path
import bm25s, numpy as np, torch
from sentence_transformers import SentenceTransformer

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
B = ROOT / "corpus_md_build" / "data"
SUB = ROOT / "submission"
csv.field_size_limit(10**9)
OUT.parent.mkdir(parents=True, exist_ok=True)
MODEL = "intfloat/multilingual-e5-base"
W_GRID = [round(w, 1) for w in np.arange(0, 1.01, 0.1)]
N_BOOT = 10000

tok = lambda t: [w for w in re.split(r"[\s\.,;:!?()\[\]\"'“”‘’।|/\\\-–—]+", t.lower()) if w]

# ---------- normaliser (identical to 16v2_normaliser.py) ----------
BN = {}
for chars, s in {"কখ": "k", "গঘ": "g", "চছ": "c", "জঝয": "j", "টঠতথৎ": "t", "ডঢদধ": "d", "ড়ঢ়": "r", "ঞণনঙং": "n",
                 "পফ": "p", "বভ": "b", "ম": "m", "য়": "y", "র": "r", "ল": "l", "শষস": "s", "হ": "h"}.items():
    for ch in chars:
        BN[ch] = s
src16 = (ROOT / "corpus_md_build" / "16v2_normaliser.py").read_text(encoding="utf-8")
# take the skeleton functions from the project file itself so that composed Bangla characters match byte for byte
ns = {"re": re, "C": C}
exec(src16[src16.index("BN = {}"):src16.index("ids = []")], ns)
skel_bn, skel_lat = ns["skel_bn"], ns["skel_lat"]

# ---------- data ----------
corpus = list(csv.DictReader(open(SUB / "corpus.csv", encoding="utf-8")))
ids = [r["passage_id"] for r in corpus]
texts = [r["text"] for r in corpus]
idx = {d: i for i, d in enumerate(ids)}
N = len(ids)
queries = list(csv.DictReader(open(SUB / "queries.csv", encoding="utf-8")))
gold = {r["query_id"]: idx[r["passage_id"]] for r in csv.DictReader(open(SUB / "qrels.csv", encoding="utf-8"))}

toks = [tok(t) for t in texts]
freq = C.Counter(w for t in toks for w in t)
look = C.defaultdict(list)
for w, c in freq.items():
    if re.search(r"[ঀ-৿]", w) and not re.search(r"[A-Za-z0-9]", w) and len(w) >= 2:
        s = skel_bn(w)
        if len(s) >= 2:
            look[s].append((c, w))
look = {s: [w for _, w in sorted(v, reverse=True)[:5]] for s, v in look.items()}


def expand(text, K):
    out = []
    for w in dict.fromkeys(tok(text)):
        out.append(w)
        if K and re.fullmatch(r"[a-z]+", w) and len(w) >= 3:
            out += look.get(skel_lat(w), [])[:K]
    return list(dict.fromkeys(out))


ret = bm25s.BM25()
ret.index(toks, show_progress=False)

# ---------- cached passage embeddings: keep rows of released passages ----------
full_ids = [json.loads(l)["doc_id"] for l in open(B / "corpus.jsonl", encoding="utf-8")]
row = {d: i for i, d in enumerate(full_ids)}
emb = np.load(B / "emb_e5base_passages.npy")
assert emb.shape[0] == len(full_ids)
P = torch.from_numpy(emb[[row[d] for d in ids]].astype("float32"))
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer(MODEL, device=device)

# alignment check: re-encode 24 passages and compare with the cached rows
model_len = model.max_seq_length
model.max_seq_length = 256
rng0 = random.Random(1)
pick = rng0.sample(range(N), 24)
E = model.encode(["passage: " + texts[i] for i in pick], normalize_embeddings=True, convert_to_tensor=True).cpu().float()
align = (E * P[pick]).sum(1).numpy()
model.max_seq_length = model_len
print("embedding alignment cosine: min %.5f" % align.min(), flush=True)
assert align.min() > 0.99


# ---------- scoring ----------
def mm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-9)


def rank_of(scores, gi):
    """0-based rank of item gi; ties count against the system."""
    g = scores[gi]
    return int((scores > g).sum() + (scores == g).sum() - 1)


def metrics(r):
    return {"nDCG@10": 1 / math.log2(r + 2) if r is not None and r < 10 else 0.0,
            "R@10": float(r is not None and r < 10),
            "R@100": float(r is not None and r < 100),
            "MRR@100": 1 / (r + 1) if r is not None and r < 100 else 0.0}


# unit checks of the metric code
_s = np.array([3.0, 2.0, 2.0, 1.0])
assert rank_of(_s, 0) == 0 and rank_of(_s, 1) == 2 and rank_of(np.zeros(5), 0) == 4
assert abs(metrics(2)["nDCG@10"] - 0.5) < 1e-12 and metrics(10)["R@10"] == 0 and metrics(10)["MRR@100"] == 1 / 11


def bm_scores(t):
    docs, sc = ret.retrieve([t], k=1000, show_progress=False)
    b = np.zeros(N, dtype="float32")
    b[docs[0]] = sc[0]
    return b, docs[0]


def full_rank(scores, g):
    # a passage with BM25 score 0 shares no term with the query: not retrieved
    if scores[g] <= 0:
        return None
    return rank_of(scores, g)


encoded = {}


def encode(qs):
    todo = [q for q in qs if q["query_id"] not in encoded]
    if todo:
        Q = model.encode(["query: " + q["query"] for q in todo], batch_size=64, normalize_embeddings=True,
                         convert_to_tensor=True).cpu().float()
        for q, v in zip(todo, Q):
            encoded[q["query_id"]] = v


def prepare(q, d, K_list):
    g = gold[q["query_id"]]
    od = np.argpartition(-d, 1000)[:1000]
    lanes = {}
    for name, terms in [("raw", tok(q["query"]))] + [(f"K{K}", expand(q["query"], K)) for K in K_list]:
        b, top = bm_scores(terms)
        cand = np.array(sorted(set(top.tolist()) | set(od.tolist())))
        w = np.where(cand == g)[0]
        lanes[name] = dict(b=b, top=top, cand=cand, gi=int(w[0]) if len(w) else None,
                           dm=mm(d[cand]), bm=mm(b[cand]))
    return g, od, lanes


def convex_rank(lane, w):
    if lane["gi"] is None:
        return None
    return rank_of(w * lane["dm"] + (1 - w) * lane["bm"], lane["gi"])


def rrf_rank(lane, d, od, g, drop_zero=False):
    top = lane["top"]
    b = lane["b"]
    border = top[np.argsort(-b[top], kind="stable")]
    if drop_zero:
        border = border[b[border] > 0]
    dorder = od[np.argsort(-d[od], kind="stable")]
    rb = {int(i): r for r, i in enumerate(border)}
    rd = {int(i): r for r, i in enumerate(dorder)}
    keys = sorted(set(rb) | set(rd))
    sc = np.array([1 / (60 + rb.get(i, 1000)) + 1 / (60 + rd.get(i, 1000)) for i in keys])
    if g not in rb and g not in rd:
        return None
    return rank_of(sc, keys.index(g))


def batches(qs, n=256):
    for s in range(0, len(qs), n):
        ch = qs[s:s + n]
        encode(ch)
        D = (torch.stack([encoded[q["query_id"]] for q in ch]) @ P.T).numpy()
        for q, d in zip(ch, D):
            yield q, d


part = lambda q: q["query_id"][0]  # P = parallel, G = non-parallel

# ---------- fusion weights from training queries only ----------
weights, curves, train_n, single = {}, {}, {}, {}
for pt in ("P", "G"):
    tr = [q for q in queries if part(q) == pt and q["split"] == "train"]
    random.seed(42)
    random.shuffle(tr)
    tr = tr[:2500]
    acc = C.defaultdict(lambda: C.defaultdict(list))
    for q, d in batches(tr):
        _, _, lanes = prepare(q, d, [3])
        for w in W_GRID:
            acc[q["script"]][w].append(metrics(convex_rank(lanes["K3"], w))["nDCG@10"])
    curves[pt] = {s: {str(w): float(np.mean(acc[s][w])) for w in W_GRID} for s in acc}
    train_n[pt] = {s: len(acc[s][0.0]) for s in acc}
    weights[pt] = {s: max(W_GRID, key=lambda w: curves[pt][s][str(w)]) for s in acc}
    allv = {w: float(np.mean([v for s in acc for v in acc[s][w]])) for w in W_GRID}
    curves[pt]["all"] = {str(w): allv[w] for w in W_GRID}
    single[pt] = max(W_GRID, key=lambda w: allv[w])
    print("weights", pt, weights[pt], "single", single[pt], train_n[pt], flush=True)

# ---------- dev and test ----------
SYS = ["bm25", "dense", "rrf", "convex0.3", "convex0.5", "convex0.7",
       "bm25_norm_K1", "bm25_norm_K3", "bm25_norm_K5", "convex0.7_norm", "rrf_norm", "adaptive", "rrf_nozero", "tuned_single"]
ev = [q for q in queries if q["split"] in ("dev", "test")]
per = {}
for n_done, (q, d) in enumerate(batches(ev), 1):
    g, od, L = prepare(q, d, [1, 3, 5])
    qid = q["query_id"]
    r = {}
    r["bm25"] = full_rank(L["raw"]["b"], g)
    for K in (1, 3, 5):
        r[f"bm25_norm_K{K}"] = full_rank(L[f"K{K}"]["b"], g)
    r["dense"] = rank_of(d, g)
    for w in (0.3, 0.5, 0.7):
        r[f"convex{w}"] = convex_rank(L["raw"], w)
    r["convex0.7_norm"] = convex_rank(L["K3"], 0.7)
    r["adaptive"] = convex_rank(L["K3"], weights[part(q)][q["script"]])
    r["tuned_single"] = convex_rank(L["K3"], single[part(q)])
    r["rrf"] = rrf_rank(L["raw"], d, od, g)
    r["rrf_norm"] = rrf_rank(L["K3"], d, od, g)
    r["rrf_nozero"] = rrf_rank(L["raw"], d, od, g, drop_zero=True)
    qt = set(tok(q["query"]))
    per[qid] = dict(part=part(q), split=q["split"], script=q["script"], domain=q["domain"], nw=len(q["query"].split()),
                    overlap=len(qt & set(toks[g])) / max(1, len(qt)),
                    ranks=r, in_pool=L["K3"]["gi"] is not None, m={s: metrics(r[s]) for s in SYS})
    if n_done % 512 == 0:
        print(n_done, "/", len(ev), flush=True)


def sel(**kw):
    return [i for i, x in per.items() if all(x[k] in (v if isinstance(v, tuple) else (v,)) for k, v in kw.items())]


def mean(ids_, s, m="nDCG@10"):
    return float(np.mean([per[i]["m"][s][m] for i in ids_]))


SCR = ("bangla", "mixed", "romanized")
table = {}
for pt in ("P", "G", "ALL"):
    for sp in ("dev", "test"):
        for sc in SCR + ("ALL",):
            kw = dict(split=sp)
            if pt != "ALL":
                kw["part"] = pt
            if sc != "ALL":
                kw["script"] = sc
            I = sel(**kw)
            table[f"{pt}|{sp}|{sc}"] = {"n": len(I), **{s: {m: mean(I, s, m) for m in ("nDCG@10", "R@10", "R@100", "MRR@100")} for s in SYS}}

rng = np.random.default_rng(42)


def boot(diff):
    diff = np.asarray(diff)
    idxs = rng.integers(0, len(diff), (N_BOOT, len(diff)))
    means = diff[idxs].mean(1)
    return {"diff": float(diff.mean()), "lo": float(np.percentile(means, 2.5)), "hi": float(np.percentile(means, 97.5)), "n": len(diff)}


pairs = [("bm25_norm_K3", "bm25"), ("convex0.7_norm", "convex0.7"), ("convex0.7_norm", "dense"), ("rrf_norm", "rrf"),
         ("adaptive", "convex0.7_norm"), ("adaptive", "dense"), ("convex0.7", "dense"),
         ("adaptive", "tuned_single"), ("adaptive", "rrf_norm")]
paired = {}
for pt in ("P", "G"):
    for sps, label in ((("test",), "test"), (("dev",), "dev"), (("dev", "test"), "devtest")):
        for sc in SCR + ("ALL",):
            kw = dict(part=pt, split=sps)
            if sc != "ALL":
                kw["script"] = sc
            I = sel(**kw)
            for a, b in pairs:
                paired[f"{pt}|{label}|{sc}|{a}-{b}"] = boot([per[i]["m"][a]["nDCG@10"] - per[i]["m"][b]["nDCG@10"] for i in I])

# script gap on the parallel part, paired by information need
gaps = {}
needs = sorted({i[:-2] for i in sel(part="P", split="test")})
for s in ["bm25", "dense", "convex0.7", "convex0.7_norm", "adaptive"]:
    for a, b in (("b", "m"), ("b", "r"), ("m", "r")):
        have = [n for n in needs if f"{n}_{a}" in per and f"{n}_{b}" in per]
        gaps[f"{s}|{a}-{b}"] = boot([per[f"{n}_{a}"]["m"][s]["nDCG@10"] - per[f"{n}_{b}"]["m"][s]["nDCG@10"] for n in have])

# generator of every parallel passage
gen_of = {}
for fn, default in (("parallel_cache.jsonl", "claude-haiku-5-5"), ("parallel_cache_groq.jsonl", "openai/gpt-oss-120b")):
    for l in open(B / fn, encoding="utf-8"):
        r = json.loads(l)
        gen_of.setdefault(r["doc_id"], r.get("generator", default))
bygen = {}
for gname in sorted(set(gen_of.values())):
    for sp in ("test", "devtest"):
        for sc in SCR:
            I = [i for i in sel(part="P", split=("test",) if sp == "test" else ("dev", "test"), script=sc) if gen_of[ids[gold[i]]] == gname]
            if I:
                bygen[f"{gname}|{sp}|{sc}"] = {"n": len(I), **{s: mean(I, s) for s in ("bm25", "dense", "convex0.7_norm", "adaptive")}}
pgen = C.Counter(gen_of[ids[gold[q["query_id"]]]] for q in queries if part(q) == "P" and q["query_id"].endswith("_b"))

# overlap, rank headroom, length and domain
extra = {}
for pt in ("P", "G"):
    for sc in SCR:
        I = sel(part=pt, split="test", script=sc)
        rk = [per[i]["ranks"]["adaptive"] for i in I]
        extra[f"{pt}|{sc}"] = {"n": len(I), "overlap": float(np.mean([per[i]["overlap"] for i in I])),
                               "top10": sum(x is not None and x < 10 for x in rk) / len(I),
                               "top100": sum(x is not None and x < 100 for x in rk) / len(I),
                               "pool": sum(per[i]["in_pool"] for i in I) / len(I)}
groups = {}
I = sel(part="P", split="test")
for lab, lo, hi in (("<=7", 0, 7), ("8-11", 8, 11), (">=12", 12, 999)):
    J = [i for i in I if lo <= per[i]["nw"] <= hi]
    groups[f"length|{lab}"] = {"n": len(J), **{s: mean(J, s) for s in ("bm25", "dense", "adaptive")}}
for dom, n in C.Counter(per[i]["domain"] for i in I).most_common():
    if n >= 30:
        J = [i for i in I if per[i]["domain"] == dom]
        groups[f"domain|{dom}"] = {"n": len(J), **{s: mean(J, s) for s in ("bm25", "dense", "adaptive")}}

# reviewed Romanized queries of the G part: text before and after human correction (all splits, no tuned weight used)
rev = {}
relq = {q["query_id"]: q for q in queries}
rv = []
for l in open(B / "queries.jsonl", encoding="utf-8"):
    q = json.loads(l)
    m = q["metadata"]
    if q["_id"] in relq and m["script"] == "romanized" and "human_reviewed" in m["source"]:
        rv.append((q["_id"], m.get("text_original", q["text"]), relq[q["_id"]]["query"]))
rev["n"] = len(rv)
rev["changed"] = sum(a != b for _, a, b in rv)
for ver, col in (("original", 1), ("released", 2)):
    tmp = [{"query_id": x[0], "query": x[col]} for x in rv]
    for x in tmp:
        encoded.pop(x["query_id"], None)
    out = C.defaultdict(list)
    for q, d in batches(tmp):
        g, od, L = prepare(q, d, [3])
        out["bm25"].append(metrics(full_rank(L["raw"]["b"], g))["nDCG@10"])
        out["bm25_norm_K3"].append(metrics(full_rank(L["K3"]["b"], g))["nDCG@10"])
        out["dense"].append(metrics(rank_of(d, g))["nDCG@10"])
        out["convex0.7"].append(metrics(convex_rank(L["raw"], 0.7))["nDCG@10"])
        out["convex0.7_norm"].append(metrics(convex_rank(L["K3"], 0.7))["nDCG@10"])
    rev[ver] = {k: float(np.mean(v)) for k, v in out.items()}
    rev[ver + "_gain"] = boot(np.array(out["convex0.7_norm"]) - np.array(out["convex0.7"]))

json.dump(dict(n_corpus=N, n_queries=len(queries), skeletons=len(look), align_min=float(align.min()), weights=weights,
               train_n=train_n, curves=curves, single=single, table=table, paired=paired, script_gaps=gaps, by_generator=bygen,
               parallel_passages_by_generator=dict(pgen), extra=extra, groups=groups, reviewed=rev),
          open(OUT, "w"), indent=1, ensure_ascii=False)
json.dump({i: {"part": x["part"], "split": x["split"], "script": x["script"], "ranks": x["ranks"]} for i, x in per.items()},
          open(OUT.with_name("per_query_ranks.json"), "w"))
print("written", OUT)
