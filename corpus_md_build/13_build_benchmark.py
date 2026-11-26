import csv, hashlib, json, re, collections as C
bn = lambda s: len(re.findall(r"[ঀ-৿]", s)); la = lambda s: len(re.findall(r"[A-Za-z]", s))
def script(q):
    b, l = bn(q), la(q)
    return "bangla" if b and l / (b + l) < 0.05 else "romanized" if not b and l else "mixed" if b and l else "other"
norm = lambda q: re.sub(r"[\s\W_]+", " ", q.lower()).strip()
rows = [json.loads(l) for l in open("data/gen_cache.jsonl", encoding="utf-8")]
seen = {}; qs = []; stat = C.Counter()
for r in rows:
    for style, q in r["kept"].items():
        q = q.strip(); sc = script(q); stat["generated"] += 1
        want = {"bangla": "bangla", "romanized": "romanized", "code_mixed": "mixed"}[style]
        if sc != want: stat["script_mismatch_dropped"] += 1; continue
        k = norm(q)
        if len(k.split()) < 3: stat["too_short_dropped"] += 1; continue
        if k in seen: stat["duplicate_dropped"] += 1; continue
        seen[k] = 1
        qs.append((r, style, q, sc))
def split(doc):  # all queries of one passage stay together
    h = int(hashlib.md5(doc.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else "dev" if h < 85 else "test"
n = 0
with open("data/queries.jsonl", "w", encoding="utf-8") as fq, open("data/qrels.csv", "w", newline="", encoding="utf-8") as fr:
    w = csv.writer(fr); w.writerow(["query-id", "corpus-id", "score"])
    for r, style, q, sc in qs:
        n += 1; qid = f"G{n:05d}"
        fq.write(json.dumps({"_id": qid, "text": q, "metadata": {"script": sc, "domain": r["domain"], "source": "llm_generated_unreviewed",
                 "split": split(r["doc_id"]), "source_doc": r["doc_id"]}}, ensure_ascii=False) + "\n")
        w.writerow([qid, r["doc_id"], 2])
print(dict(stat), "| final queries", n)
sc = C.Counter(q[3] for q in qs); print("script", dict(sc))
print("split", C.Counter(split(q[0]["doc_id"]) for q in qs))
print("domain", C.Counter(q[0]["domain"] for q in qs).most_common())
print("passages", len({q[0]["doc_id"] for q in qs}))
