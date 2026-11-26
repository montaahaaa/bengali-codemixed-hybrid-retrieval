"""Re-grade REAL Romanized and mixed-script queries that the first grading could not reach (issue: answerable yield, script balance).
Usage: python scripts/regrade_romanized.py
Why: the first pass used plain BM25 candidates, which cannot reach Bangla passages from Latin-script queries (D-035), so those
queries were graded 0 for lack of candidates, not for lack of an answer. Here the candidates come from BM25 with the
consonant-skeleton expansion of Experiment 2 (K=3, top 30), then bge-reranker-v2-m3 grades the best passage
(>=0.60 -> 2, >=0.20 -> 1, uncalibrated, same as rerank_grade.py).
Queries: source youtube/facebook, script latin or mixed, no relevance row yet in qrels. Output (private text, git-ignored):
data/processed/regrade_romanized.csv. Grades are a PROXY; the owner must confirm any grade-2 before it is counted."""
import csv, re, sys, time, array, collections as C
from pathlib import Path
import numpy as np, scipy.sparse as sp, torch
from sentence_transformers import CrossEncoder
sys.path.insert(0, str(Path(__file__).resolve().parent))
from find_candidates import load, tok
ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/benchmark_full/"
TOP = 30
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

bm, ids = load(); pid = [x[0] for x in ids]
vocab = {w: i for i, w in enumerate(bm.idf)}
look = C.defaultdict(list)
for w, i in vocab.items():
    if re.search(r"[ঀ-৿]", w) and not re.search(r"[A-Za-z0-9]", w) and len(w) >= 2:
        s = skel_bn(w)
        if len(s) >= 2: look[s].append((bm.idf[w], w))
look = {s: [w for _, w in sorted(v)[:5]] for s, v in look.items()}
k1, b, avgdl = bm.k1, bm.b, bm.avgdl
ptr, ind, dat = [0], array.array("i"), array.array("f")
for freqs, dl in zip(bm.doc_freqs, bm.doc_len):
    norm = k1 * (1 - b + b * dl / avgdl)
    for w, tf in freqs.items(): ind.append(vocab[w]); dat.append(bm.idf[w] * tf * (k1 + 1) / (tf + norm))
    ptr.append(len(ind))
X = sp.csr_matrix((np.frombuffer(dat, np.float32), np.frombuffer(ind, np.int32), np.array(ptr, np.int64)), shape=(len(pid), len(vocab))).tocsc()
print("index ready", flush=True)
def expand(text, K=3):
    out = []
    for w in dict.fromkeys(tok(text)):
        out.append(w)
        if re.fullmatch(r"[a-z]+", w) and len(w) >= 3: out += look.get(skel_lat(w), [])[:K]
    return list(dict.fromkeys(out))

q = list(csv.DictReader(open(D / "queries.csv", encoding="utf-8-sig")))
have = {l.split()[0] for l in open(D / "qrels.tsv")}
sel = [r for r in q if r["source"] in ("youtube", "facebook") and r["script"] in ("latin", "mixed")
       and r["qid"] not in have and len(tok(r["query"])) >= 3]
print(len(sel), "queries to grade", flush=True)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = CrossEncoder("BAAI/bge-reranker-v2-m3", device=dev, max_length=320, activation_fn=torch.nn.Sigmoid(),
                     model_kwargs={"torch_dtype": torch.float16})
t0 = time.time()
out = ROOT / "data/processed/regrade_romanized.csv"
with out.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["qid", "source", "script", "domain", "query", "expanded_tokens", "grade", "score", "second_score", "passage_id", "title", "passage_text", "owner_confirms_yes_no", "notes"])
    for k, r in enumerate(sel, 1):
        toks = [vocab[x] for x in expand(r["query"]) if x in vocab]
        if not toks: w.writerow([r["qid"], r["source"], r["script"], r["domain"], r["query"], 0, 0, "0.000", "0.000", "", "", "", "", ""]); continue
        s = np.asarray(X[:, toks].sum(axis=1)).ravel()
        top = np.argpartition(-s, TOP)[:TOP]; top = top[np.argsort(-s[top])]
        top = [i for i in top if s[i] > 0]
        if not top: w.writerow([r["qid"], r["source"], r["script"], r["domain"], r["query"], len(toks), 0, "0.000", "0.000", "", "", "", "", ""]); continue
        sc = model.predict([(r["query"], f"{ids[i][1]}. {ids[i][2]}") for i in top], batch_size=15, show_progress_bar=False)
        order = np.argsort(-sc); bi = order[0]; g = 2 if sc[bi] >= 0.60 else 1 if sc[bi] >= 0.20 else 0
        i = top[bi]
        w.writerow([r["qid"], r["source"], r["script"], r["domain"], r["query"], len(toks), g, f"{sc[bi]:.3f}",
                    f"{sc[order[1]]:.3f}" if len(order) > 1 else "0.000", ids[i][0], ids[i][1], ids[i][2][:500], "", ""])
        if k % 100 == 0: print(f"{k}/{len(sel)} {time.time()-t0:.0f}s", flush=True); f.flush()
print("done", flush=True)
