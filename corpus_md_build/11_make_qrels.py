import csv, json, collections as C
meta = {}
for l in open("data/queries.jsonl", encoding="utf-8"):
    r = json.loads(l); meta[r["_id"]] = r["metadata"]
rows = []; best = {}
for l in open("data/rerank_scores.jsonl"):
    r = json.loads(l); s = r["scores"]; b = max(s.values()); best[r["qid"]] = b
    for d, v in s.items():
        g = 2 if v >= 0.60 else 1 if v >= 0.20 else 0
        if g: rows.append((r["qid"], d, g))
with open("data/qrels.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["query-id", "corpus-id", "score"]); w.writerows(rows)
print("graded queries", len(best), "| qrels rows", len(rows))
def tier(b): return "answered(>=0.6)" if b >= .6 else "weak(0.2-0.6)" if b >= .2 else "none(<0.2)"
for key in ("script", "domain"):
    t = C.defaultdict(C.Counter)
    for q, b in best.items(): t[meta[q][key]][tier(b)] += 1
    print(f"\nby {key}:")
    for k, c in sorted(t.items()): n = sum(c.values()); print(f"  {k:24s} n={n:5d}  answered {c['answered(>=0.6)']/n:5.1%}  weak {c['weak(0.2-0.6)']/n:5.1%}  none {c['none(<0.2)']/n:5.1%}")
