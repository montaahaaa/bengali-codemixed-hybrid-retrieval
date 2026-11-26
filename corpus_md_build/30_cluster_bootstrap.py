"""Usage (from corpus_md_build): python 30_cluster_bootstrap.py data/release_eval ..
Passage-clustered paired bootstrap for rows that pool the three scripts (up to three queries share a gold passage)."""
import csv, json, math, sys, collections as C
from pathlib import Path
import numpy as np
S, R = Path(sys.argv[1]), Path(sys.argv[2])
per = json.load(open(S / "per_query_ranks.json"))
gold = {r["query_id"]: r["passage_id"] for r in csv.DictReader(open(R / "submission" / "qrels.csv"))}
nd = lambda r: 1 / math.log2(r + 2) if r is not None and r < 10 else 0.0
rng = np.random.default_rng(42); out = {}
pairs = [("bm25_norm_K3", "bm25"), ("convex0.7_norm", "convex0.7"), ("convex0.7_norm", "dense"), ("rrf_norm", "rrf"),
         ("adaptive", "convex0.7_norm"), ("adaptive", "dense"), ("convex0.7", "dense"), ("adaptive", "tuned_single"), ("adaptive", "rrf_norm")]
for pt in "PG":
    for label, sps in (("test", ("test",)), ("devtest", ("dev", "test"))):
        I = [i for i, x in per.items() if x["part"] == pt and x["split"] in sps]
        for a, b in pairs:
            cl = C.defaultdict(list)
            for i in I: cl[gold[i]].append(nd(per[i]["ranks"][a]) - nd(per[i]["ranks"][b]))
            s = np.array([sum(v) for v in cl.values()]); n = np.array([len(v) for v in cl.values()])
            idx = rng.integers(0, len(s), (10000, len(s))); m = s[idx].sum(1) / n[idx].sum(1)
            out[f"{pt}|{label}|ALL|{a}-{b}"] = dict(diff=float(s.sum() / n.sum()), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)), n=int(n.sum()), passages=len(s))
            if label == "test": print(pt, a, "-", b, "%+.3f [%+.3f,%+.3f]" % (out[f"{pt}|{label}|ALL|{a}-{b}"]["diff"], out[f"{pt}|{label}|ALL|{a}-{b}"]["lo"], out[f"{pt}|{label}|ALL|{a}-{b}"]["hi"]), len(s))
json.dump(out, open(S / "cluster_boot.json", "w"), indent=1)
