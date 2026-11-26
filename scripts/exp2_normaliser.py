"""Experiment 2 (pre-defence): consonant-skeleton normaliser for Romanized Bangla (RQ3 pilot).
Usage: python scripts/exp2_normaliser.py
Idea: BM25 cannot match a Latin-script Bangla word to its Bangla-script form. Build a lookup from the corpus itself:
every Bangla word in the BM25 vocabulary gets a consonant skeleton (vowels dropped, similar consonants merged, e.g.
জাতীয় -> jty, 'jatiyo' -> jty). A Latin query token is expanded with the K most frequent Bangla words that share its skeleton.
The original Latin token is kept (so English words still match English passages). No training and no query data are used to build the lookup.
K = 3 is fixed in advance; K = 1 and 5 are reported as a sensitivity check only.
Same query sets, index and metrics as Exp 1 (synthetic = controlled, real = biased towards BM25)."""
import csv, json, re, sys, time, array, datetime, collections as C
from pathlib import Path
import numpy as np, scipy.sparse as sp
sys.path.insert(0, str(Path(__file__).resolve().parent))
from find_candidates import load, tok
ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/benchmark_full/"
K0, SEED, KS = 100, 42, (1, 3, 5)

BN = {}
for chars, s in {"কখ": "k", "গঘ": "g", "চছ": "c", "জঝয": "j", "টঠতথৎ": "t", "ডঢদধ": "d", "ড়ঢ়": "r", "ঞণনঙং": "n",
                 "পফ": "p", "বভ": "b", "ম": "m", "য়": "y", "র": "r", "ল": "l", "শষস": "s", "হ": "h"}.items():
    for ch in chars: BN[ch] = s
def skel_bn(w):
    w = w.replace("ড়", "ড়").replace("ঢ়", "ঢ়").replace("য়", "য়")
    out = []
    for i, ch in enumerate(w):
        if ch in BN and not (ch == "য" and i > 0 and w[i-1] == "্"): out.append(BN[ch])
    return re.sub(r"(.)\1+", r"\1", "".join(out))
def skel_lat(w):
    w = w.lower()
    for a, b in (("sh", "s"), ("ng", "n"), ("kh", "k"), ("gh", "g"), ("ph", "p"), ("bh", "b"), ("th", "t"), ("dh", "d"), ("jh", "j"), ("ch", "c")):
        w = w.replace(a, b)
    m = {"v": "b", "w": "b", "z": "j", "q": "k", "f": "p", "x": "k"}
    out = [m.get(ch, ch) for ch in w if ch in "bcdfghjklmnpqrstvwxyz"]
    return re.sub(r"(.)\1+", r"\1", "".join(out))

bm, ids = load(); pid = [x[0] for x in ids]; inc = set(pid)
vocab = {w: i for i, w in enumerate(bm.idf)}
t0 = time.time()
look = C.defaultdict(list)
for w, i in vocab.items():
    if re.search(r"[ঀ-৿]", w) and not re.search(r"[A-Za-z0-9]", w) and len(w) >= 2:
        s = skel_bn(w)
        if len(s) >= 2: look[s].append((bm.idf[w], w))
look = {s: [w for _, w in sorted(v)[:5]] for s, v in look.items()}
print("lookup", len(look), f"{time.time()-t0:.0f}s", flush=True)

k1, b, avgdl = bm.k1, bm.b, bm.avgdl
ptr, ind, dat = [0], array.array("i"), array.array("f")
for freqs, dl in zip(bm.doc_freqs, bm.doc_len):
    norm = k1 * (1 - b + b * dl / avgdl)
    for w, tf in freqs.items(): ind.append(vocab[w]); dat.append(bm.idf[w] * tf * (k1 + 1) / (tf + norm))
    ptr.append(len(ind))
X = sp.csr_matrix((np.frombuffer(dat, np.float32), np.frombuffer(ind, np.int32), np.array(ptr, np.int64)), shape=(len(pid), len(vocab))).tocsc()
print("index ready", flush=True)

q = {r["qid"]: r for r in csv.DictReader(open(D / "queries.csv", encoding="utf-8-sig"))}
qr = C.defaultdict(dict)
for l in open(D / "qrels.tsv"):
    a, _, p, g = l.split(); qr[a][p] = int(g)
sel = [a for a in qr if q[a]["source"] != "miracl_bn" and any(p in inc for p in qr[a])]
def expand(text, K):
    out = []
    for w in dict.fromkeys(tok(text)):
        out.append(w)
        if re.fullmatch(r"[a-z]+", w) and len(w) >= 3:
            out += look.get(skel_lat(w), [])[:K]
    return list(dict.fromkeys(out))
def metrics(qid, K):
    toks = [vocab[w] for w in (expand(q[qid]["query"], K) if K else dict.fromkeys(tok(q[qid]["query"]))) if w in vocab]
    rel = {p: g for p, g in qr[qid].items() if p in inc and g >= 1}
    if not toks: return dict(r10=0, r100=0, mrr=0, ndcg=0)
    s = np.asarray(X[:, toks].sum(axis=1)).ravel()
    top = np.argpartition(-s, K0)[:K0]; top = top[np.argsort(-s[top])]
    ranked = [pid[i] for i in top if s[i] > 0]; nrel = len(rel)
    hit = [i for i, p in enumerate(ranked) if p in rel]
    dcg = sum(rel[p] / np.log2(i + 2) for i, p in enumerate(ranked[:10]) if p in rel)
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(sorted(rel.values(), reverse=True)[:10]))
    return dict(r10=len([i for i in hit if i < 10]) / nrel, r100=len(hit) / nrel, mrr=(1 / (hit[0] + 1) if hit and hit[0] < 10 else 0), ndcg=dcg / idcg if idcg else 0)
res = {}
for K in (0,) + KS:
    for k, a in enumerate(sel, 1): res[(K, a)] = metrics(a, K)
    print("K", K, "done", f"{time.time()-t0:.0f}s", flush=True)
rng = np.random.default_rng(SEED)
def agg(K, qs, m):
    v = np.array([res[(K, a)][m] for a in qs]); bs = [rng.choice(v, len(v)).mean() for _ in range(1000)]
    return round(float(v.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)
out = {}
for name, f in (("synthetic", lambda a: q[a]["source"].startswith("llm_")), ("real", lambda a: not q[a]["source"].startswith("llm_"))):
    for scr in ("bn", "mixed", "latin"):
        qs = [a for a in sel if f(a) and q[a]["script"] == scr]
        for K in (0,) + KS: out[f"{name}/{scr}/K={K}"] = dict(n=len(qs), **{m: agg(K, qs, m) for m in ("r10", "r100", "mrr", "ndcg")})
# paired bootstrap on synthetic latin: K=3 minus baseline, nDCG@10
qs = [a for a in sel if q[a]["source"].startswith("llm_") and q[a]["script"] == "latin"]
d = np.array([res[(3, a)]["ndcg"] - res[(0, a)]["ndcg"] for a in qs]); bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
out["paired_synthetic_latin_ndcg_K3_minus_base"] = [round(float(d.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
json.dump(out, open(ROOT / "experiments/exp2_normaliser_results.json", "w"), indent=1)
with open(ROOT / "experiments/experiment_tracker.csv", "a", newline="") as f:
    w = csv.writer(f)
    for k, v in out.items():
        if isinstance(v, dict):
            for m in ("r10", "r100", "mrr", "ndcg"): w.writerow([f"exp2_normaliser/{k}", datetime.date.today().isoformat(), f"BM25 + skeleton lookup, n={v['n']}", m, v[m][0]])
print("done")
