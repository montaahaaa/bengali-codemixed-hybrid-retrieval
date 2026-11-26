"""Finalize v1 of the generated benchmark: (3) apply the human Romanized review, (4) remove near-duplicate queries. Originals in data/backup_pre_finalize/."""
import csv, json, re, collections as C
import numpy as np, scipy.sparse as sp
THR = 0.7
qs = [json.loads(l) for l in open("data/backup_pre_finalize/queries.jsonl", encoding="utf-8")]
rows = list(csv.reader(open("data/backup_pre_finalize/qrels.csv"))); hdr, qr = rows[0], rows[1:]
rv = {r["q_romanized_id"]: r for r in csv.DictReader(open("data/review/romanized_review_results.csv", encoding="utf-8-sig"))}
stat = C.Counter()
for q in qs:                                                    # step 3: apply the review
    m = q["metadata"]; m["review"] = {}
    r = rv.get(q["_id"])
    if r:
        if r["verdict"] == "change":
            m["text_original"] = q["text"]; q["text"] = r["reviewed_text"].strip(); m["review"]["romanized"] = "corrected"; stat["corrected"] += 1
        else: m["review"]["romanized"] = "verified_no_change"; stat["verified"] += 1
        m["source"] = "llm_generated_human_reviewed_romanized"
    else: m["source"] = "llm_generated_unreviewed"
tokz = lambda t: set(w for w in re.split(r"[\s\.,;:!?()\[\]\"'“”‘’।|/\\\-–—]+", t.lower()) if w)
keep, drop, ex = [], {}, []                                      # step 4: near duplicates (word Jaccard >= THR, same script)
for script in ("bangla", "mixed", "romanized"):
    g = [q for q in qs if q["metadata"]["script"] == script]
    g.sort(key=lambda q: (q["metadata"]["source"] == "llm_generated_unreviewed", q["_id"]))   # reviewed first
    T = [tokz(q["text"]) for q in g]; voc = {}
    for t in T:
        for w in t: voc.setdefault(w, len(voc))
    X = sp.lil_matrix((len(g), len(voc)), dtype=np.float32)
    for i, t in enumerate(T):
        for w in t: X[i, voc[w]] = 1
    X = X.tocsr(); I = (X @ X.T).tocoo(); n = np.asarray(X.sum(axis=1)).ravel()
    nbrs = C.defaultdict(list)
    for i, j, v in zip(I.row, I.col, I.data):
        if i < j and n[i] >= 4 and n[j] >= 4 and v / (n[i] + n[j] - v) >= THR: nbrs[j].append(i); nbrs[i].append(j)
    kept = set()
    for i in range(len(g)):
        dup = [k for k in nbrs[i] if k in kept]
        if dup and g[i]["metadata"]["source"] == "llm_generated_unreviewed":
            drop[g[i]["_id"]] = g[dup[0]]["_id"]; ex.append((g[dup[0]]["text"], g[i]["text"]))
        else: kept.add(i)
stat["near_duplicates_removed"] = len(drop)
out = [q for q in qs if q["_id"] not in drop]
with open("data/queries.jsonl", "w", encoding="utf-8") as f:
    for q in out: f.write(json.dumps(q, ensure_ascii=False) + "\n")
with open("data/qrels.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(hdr); w.writerows(r for r in qr if r[0] not in drop)
json.dump(drop, open("data/near_duplicates_removed.json", "w"), indent=1)
print(dict(stat), "| queries", len(qs), "->", len(out))
for a, b in ex[:6]: print("  KEPT:", a, "\n  DROP:", b)
