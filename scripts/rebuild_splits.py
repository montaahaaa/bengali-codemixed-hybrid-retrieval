"""Rebuild train/dev/test splits for data/benchmark_full (advisor audit, D-036/D-037).

Writes data/benchmark_full/splits_v2.csv (qid, split_v2, group_id, group_size, domain_supervised).
Does not touch queries.csv, so the old split stays reproducible.

Rules:
- Queries are grouped so that near-copies never straddle splits:
  (a) same text after NFKC, lowercase, punctuation/space removal;
  (b) word Jaccard >= 0.8 on queries with >= 4 words;
  (c) same grade-2 answer passage (non-MIRACL only, groups capped, see MAXG).
- Groups are split 70/10/20 train/dev/test, stratified by (source, script, answerable).
- LLM-synthetic queries stay in their own split, never train/dev/test; a synthetic
  query grouped with a real one is "excluded".
- domain_supervised = 0 for the 391 `unassigned` queries.
"""
import csv, re, random, unicodedata, collections as C, sys
D = "data/benchmark_full/"
SEED, MAXG = 42, 50
q = list(csv.DictReader(open(D + "queries.csv", encoding="utf-8-sig")))
tier = {r["qid"]: r["tier"] for r in csv.DictReader(open(D + "benchmark_flat.csv", encoding="utf-8-sig"))}
norm = lambda s: re.sub(r"\W+", " ", unicodedata.normalize("NFKC", s).lower()).strip()

par = {r["qid"]: r["qid"] for r in q}
def find(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
def union(a, b): par[find(a)] = find(b)

by_norm = C.defaultdict(list)
for r in q: by_norm[norm(r["query"])].append(r["qid"])
for v in by_norm.values():
    for x in v[1:]: union(v[0], x)

toks = [(r["qid"], set(norm(r["query"]).split())) for r in q]
toks = [(i, t) for i, t in toks if len(t) >= 4]
inv = C.defaultdict(list)
for k, (_, t) in enumerate(toks):
    for w in t: inv[w].append(k)
for k, (i, t) in enumerate(toks):
    cand = C.Counter(j for w in t if len(inv[w]) < 300 for j in inv[w] if j > k)
    for j, c in cand.items():
        if c >= 4 and len(t & toks[j][1]) / len(t | toks[j][1]) >= 0.8: union(i, toks[j][0])

src = {r["qid"]: r["source"] for r in q}
pas = C.defaultdict(list)
for l in open(D + "qrels.tsv"):
    qd, _, pid, g = l.split()
    if g == "2" and src[qd] != "miracl_bn": pas[pid].append(qd)
merged = skipped = 0
for pid, v in pas.items():
    roots = {find(x) for x in v}
    if len(roots) > 1:
        size = C.Counter(find(r["qid"]) for r in q)
        if sum(size[x] for x in roots) <= MAXG:
            for x in v[1:]: union(v[0], x); merged += 1
        else: skipped += 1

groups = C.defaultdict(list)
for r in q: groups[find(r["qid"])].append(r)
rows = {r["qid"]: r for r in q}
ans = lambda r: tier[r["qid"]] in ("core", "partial", "miracl")
strata = C.defaultdict(list)
out = {}
for g, m in groups.items():
    if all(r["split"] == "synthetic" for r in m):
        for r in m: out[r["qid"]] = "synthetic"
        continue
    key = (m[0]["source"], m[0]["script"], any(ans(r) for r in m))
    strata[key].append(g)
rng = random.Random(SEED)
for key in sorted(strata, key=str):
    gs = sorted(strata[key]); rng.shuffle(gs)
    n = len(gs); a, b = round(n * .7), round(n * .8)
    for k, g in enumerate(gs):
        for r in groups[g]:
            # a synthetic query that duplicates a real one is excluded, not trained on
            out[r["qid"]] = "excluded" if r["split"] == "synthetic" else "train" if k < a else "dev" if k < b else "test"

with open(D + "splits_v2.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["qid", "split_v2", "group_id", "group_size", "domain_supervised"])
    for r in q:
        g = find(r["qid"]); w.writerow([r["qid"], out[r["qid"]], g, len(groups[g]), int(r["domain"] != "unassigned")])

# report
print("groups", len(groups), "multi-member", sum(len(m) > 1 for m in groups.values()), "max size", max(map(len, groups.values())), "passage merges", merged, "skipped", skipped)
print(C.Counter(out.values()))
print("answerable per split", C.Counter(out[r["qid"]] for r in q if ans(r) and tier[r["qid"]] != "miracl"))
for dim in ("source", "script", "domain"):
    print(dim); t = C.Counter((r[dim], out[r["qid"]]) for r in q)
    for k in sorted({a for a, _ in t}): print(" ", k, {s: t[(k, s)] for s in ("train", "dev", "test", "synthetic") if t[(k, s)]})
leak = sum(len({out[r["qid"]] for r in m}) > 1 for m in groups.values()); print("groups spanning splits:", leak)
