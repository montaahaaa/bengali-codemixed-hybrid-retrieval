"""Step 3: build the competition CSVs in ../submission/. Re-runnable.
queries.csv = parallel v2 queries (parallel=true) + v1 queries for passages NOT in v2 (parallel=false).
Government-site passages and their queries are excluded (no open licence). Passage-level splits are kept (v2 and v1 use the same md5 rule)."""
import csv, json, os, collections as C
os.makedirs("../submission", exist_ok=True)
corpus = {}
for l in open("data/corpus.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["source_type"] == "wikipedia": corpus[r["doc_id"]] = r
v2 = [json.loads(l) for l in open("data/v2/queries.jsonl", encoding="utf-8")]
v1 = [json.loads(l) for l in open("data/queries.jsonl", encoding="utf-8")]
v2_docs = {q["metadata"]["source_doc"] for q in v2}
OV = json.load(open("data/review_overrides.json", encoding="utf-8"))  # reviewer-1 corrections
rows = []; st = C.Counter()
for q in v2:
    m = q["metadata"]
    if m["source_doc"] not in corpus: st["v2_dropped_not_wikipedia"] += 1; continue
    rows.append((q["_id"], OV.get(q["_id"], q["text"]), m["script"], m["domain"], m["split"], m["source_doc"], "true", m["need"], "llm_unreviewed"))
for q in v1:
    m = q["metadata"]
    if m["source_doc"] in v2_docs: continue
    if m["source_doc"] not in corpus: st["v1_dropped_not_wikipedia"] += 1; continue
    if m["script"] not in ("bangla", "mixed", "romanized"): st["v1_other_script"] += 1; continue
    rv = "human_reviewed" if m["source"] == "llm_generated_human_reviewed" else "llm_unreviewed"
    rows.append((q["_id"], OV.get(q["_id"], q["text"]), m["script"], m["domain"], m["split"], m["source_doc"], "false", "", rv))
seen = set(); final = []
for r in rows:
    k = (r[2], r[1].lower())
    if k in seen: st["dup_text"] += 1; continue
    seen.add(k); final.append(r)
with open("../submission/queries.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["query_id", "query", "script", "domain", "split"]); w.writerows(r[:5] for r in final)
with open("../submission/qrels.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["query_id", "passage_id", "relevance"]); [w.writerow([r[0], r[5], 1]) for r in final]
with open("../submission/corpus.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["passage_id", "title", "text", "source_url"])
    for d, r in corpus.items(): w.writerow([d, r["title"], r["text"], r["source_url"]])
st["corrections_applied"] = sum(r[0] in OV for r in final); st["queries"] = len(final); st["parallel_true"] = sum(r[6] == "true" for r in final)
for k in ("script", "split"): st.update({f"{k}={x}": n for x, n in C.Counter(r[2 if k == "script" else 4] for r in final).items()})
print(dict(st))
