import csv, hashlib, json, re, collections as C
bn = lambda s: len(re.findall(r"[ঀ-৿]", s)); la = lambda s: len(re.findall(r"[A-Za-z]", s))
def script(q):
    b, l = bn(q), la(q)
    return "bangla" if b and l / (b + l) < 0.05 else "romanized" if not b and l else "mixed" if b and l else "other"
split = lambda d: (lambda h: "train" if h < 70 else "dev" if h < 85 else "test")(int(hashlib.md5(d.encode()).hexdigest(), 16) % 100)
rows = [json.loads(l) for l in open("data/parallel_clean.jsonl", encoding="utf-8")]
stat = C.Counter(); out = []; n = 0
for r in rows:
    k = r["kept"]
    if set(k) != {"code_mixed", "romanized"}: stat["incomplete_triple_dropped"] += 1; continue
    if script(r["bangla"]) != "bangla" or script(k["code_mixed"]) != "mixed" or script(k["romanized"]) != "romanized": stat["script_mismatch_dropped"] += 1; continue
    n += 1; need = f"P{n:05d}"; stat["triples"] += 1
    for form, txt, sc in (("bangla", r["bangla"], "bangla"), ("mixed", k["code_mixed"], "mixed"), ("romanized", k["romanized"], "romanized")):
        out.append({"_id": f"{need}_{sc[0]}", "text": txt, "metadata": {"script": sc, "domain": r["domain"], "need": need, "split": split(r["doc_id"]), "source_doc": r["doc_id"],
                    "source": "llm_parallel_unreviewed", "v1_bangla_id": r["bangla_id"]}})
with open("data/v2/queries.jsonl", "w", encoding="utf-8") as f:
    for q in out: f.write(json.dumps(q, ensure_ascii=False) + "\n")
with open("data/v2/qrels.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["query-id", "corpus-id", "score"])
    for q in out: w.writerow([q["_id"], q["metadata"]["source_doc"], 2])
print(dict(stat), "queries", len(out), C.Counter(q["metadata"]["split"] for q in out))
