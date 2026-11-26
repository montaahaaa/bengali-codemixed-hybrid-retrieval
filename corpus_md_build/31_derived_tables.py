"""Tables for the single train-tuned weight (the strongest system), computed from per_query_ranks.json.
Usage (from corpus_md_build): python 31_derived_tables.py data/release_eval .."""
import csv, json, math, sys, collections as C
from pathlib import Path
import numpy as np
S, R = Path(sys.argv[1]), Path(sys.argv[2]); B = R / "corpus_md_build" / "data"
csv.field_size_limit(10**9)
per = json.load(open(S / "per_query_ranks.json"))
Q = {r["query_id"]: r for r in csv.DictReader(open(R / "submission" / "queries.csv", encoding="utf-8"))}
gold = {r["query_id"]: r["passage_id"] for r in csv.DictReader(open(R / "submission" / "qrels.csv"))}
nd = lambda i, s: (lambda r: 1 / math.log2(r + 2) if r is not None and r < 10 else 0.0)(per[i]["ranks"][s])
rng = np.random.default_rng(42)
def boot(ids, f):
    """paired difference, bootstrap over gold passages"""
    cl = C.defaultdict(list)
    for key, d in ids: cl[key].append(d)
    s = np.array([sum(v) for v in cl.values()]); n = np.array([len(v) for v in cl.values()])
    x = rng.integers(0, len(s), (10000, len(s))); m = s[x].sum(1) / n[x].sum(1)
    return dict(diff=float(s.sum() / n.sum()), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)), n=int(n.sum()))
SCR = ("bangla", "mixed", "romanized")
sel = lambda pt, sc=None: [i for i, x in per.items() if x["part"] == pt and x["split"] == "test" and (sc is None or x["script"] == sc)]
out = {"paired": {}, "gaps": {}, "head": {}, "groups": {}, "gen": {}}
for pt in "PG":
    for sc in SCR + ("ALL",):
        I = sel(pt, None if sc == "ALL" else sc)
        for a, b in (("tuned_single", "dense"), ("tuned_single", "convex0.7_norm"), ("tuned_single", "rrf_norm")):
            out["paired"][f"{pt}|{sc}|{a}-{b}"] = boot([(gold[i], nd(i, a) - nd(i, b)) for i in I], None)
    for sc in SCR:
        I = sel(pt, sc); rk = [per[i]["ranks"]["tuned_single"] for i in I]
        out["head"][f"{pt}|{sc}"] = dict(n=len(I), top10=sum(r is not None and r < 10 for r in rk) / len(I),
            top100=sum(r is not None and r < 100 for r in rk) / len(I), pool=sum(r is not None for r in rk) / len(I))
needs = sorted({i[:-2] for i in sel("P")})
for a, b in (("b", "m"), ("b", "r"), ("m", "r")):
    out["gaps"][f"{a}-{b}"] = boot([(n, nd(f"{n}_{a}", "tuned_single") - nd(f"{n}_{b}", "tuned_single")) for n in needs], None)
I = sel("P")
def agg(J): return dict(n=len(J), **{s: float(np.mean([nd(i, s) for i in J])) for s in ("bm25", "dense", "tuned_single")})
nw = {i: len(Q[i]["query"].split()) for i in I}
for lab, lo, hi in (("Up to 7 words", 0, 7), ("8 to 11 words", 8, 11), ("12 words or more", 12, 999)):
    out["groups"][lab] = agg([i for i in I if lo <= nw[i] <= hi])
for dom, n in C.Counter(Q[i]["domain"] for i in I).most_common():
    if n >= 30: out["groups"]["domain:" + dom] = agg([i for i in I if Q[i]["domain"] == dom])
gen = {}
for fn, d in (("parallel_cache.jsonl", "claude-haiku-5-5"), ("parallel_cache_groq.jsonl", "gpt-oss-120b")):
    for l in open(B / fn, encoding="utf-8"):
        r = json.loads(l); gen.setdefault(r["doc_id"], d)
for g in ("claude-haiku-5-5", "gpt-oss-120b"):
    for sc in SCR:
        J = [i for i in sel("P", sc) if gen[gold[i]] == g]; out["gen"][f"{g}|{sc}"] = agg(J)
pc = C.Counter(gen[gold[i]] for i in Q if i[0] == "P" and i.endswith("_b")); out["gen_passages"] = dict(pc)
json.dump(out, open(S / "derived.json", "w"), indent=1)
for k, v in out["paired"].items(): print(k, "%+.3f [%+.3f,%+.3f]" % (v["diff"], v["lo"], v["hi"]))
for k, v in out["gaps"].items(): print("gap", k, "%+.3f [%+.3f,%+.3f]" % (v["diff"], v["lo"], v["hi"]))
for k in ("head", "groups", "gen", "gen_passages"): print(k, json.dumps(out[k], default=float)[:1500])
