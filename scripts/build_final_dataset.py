"""Assemble the 10,000-item mining dataset and the draft benchmark. Usage: python scripts/build_final_dataset.py
Inputs (data/processed): dataset_10000_ids.json, mining_pilot.jsonl, graded_queries_rerank.csv (owner-confirmed grades),
graded_queries_rest_rerank.csv (new batch, auto grades, NOT yet checked), review labels in scripts/review_labels_batch*.py.
Outputs: data/processed/dataset_10000.csv, data/benchmark_draft/{queries.csv,qrels.tsv,passages.csv}, and counts-only
stats in data/annotations/pilot/dataset_10000_stats.json. Everything holding platform text is git-ignored (private)."""
import csv, json, random, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_labels_batch1 as b1, review_labels_batch2 as b2

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data/processed"
rd = lambda f: list(csv.DictReader(open(P / f, encoding="utf-8-sig")))
ids = json.load(open(P / "dataset_10000_ids.json"))
mine = {}
for l in open(P / "mining_pilot.jsonl", encoding="utf-8"):
    r = json.loads(l); mine[r["id"]] = r
first, rest = rd("graded_queries_rerank.csv"), rd("graded_queries_rest_rerank.csv")
graded = {r["id"]: ("owner_confirmed", r) for r in first}
graded.update({r["id"]: ("auto_unchecked", r) for r in rest})

# draft review labels for grade-2 rows
review = {}
for i, r in enumerate([x for x in rd("benchmark_draft.csv")], 1):
    review[r["query_id"]] = (b1.label(i), b1.CLEAN.get(i, ""))
for i, r in enumerate([x for x in rest if x["rerank_grade"] == "2"], 1):
    review[r["id"]] = (b2.label(i), b2.CLEAN.get(i, ""))

cols = ["id", "source", "domain", "script", "text", "question_like", "rerank_grade", "rerank_score", "best_passage_id",
        "best_title", "grade_status", "draft_review", "query_clean_draft", "manual_check", "notes"]
rows = []
for i in ids:
    m = mine[i]; st, g = graded[i]
    rv = review.get(i, ("", ""))
    rows.append([i, m["source"], m["domain"] or "", m["script"], m["text"], int(m["question_like"]), g["rerank_grade"],
                 g["rerank_score"], g["best_id"], g["best_title"], st, rv[0], rv[1], "", ""])
rows.sort(key=lambda r: (r[1], r[2], r[0]))
with open(P / "dataset_10000.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(cols); w.writerows(rows)

# draft benchmark: grade-2 rows that the draft review did not drop
random.seed(42)
bench = [r for r in rows if r[6] == "2" and r[11] in ("keep", "weak")]
random.shuffle(bench)
nd = round(0.3 * len(bench))
(ROOT / "data/benchmark_draft").mkdir(exist_ok=True, parents=True)
with open(ROOT / "data/benchmark_draft/queries.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["qid", "split", "domain", "script", "source", "query_clean_draft", "query_raw_private", "draft_review", "grade_status"])
    for k, r in enumerate(bench):
        w.writerow([f"Q{k+1:04d}", "dev" if k < nd else "test", r[2] or "unassigned", r[3], r[1], r[12], r[4], r[11], r[10]])
qid = {r[0]: f"Q{k+1:04d}" for k, r in enumerate(bench)}
with open(ROOT / "data/benchmark_draft/qrels.tsv", "w", encoding="utf-8", newline="") as f:
    for r in bench:
        f.write(f"{qid[r[0]]}\t0\t{r[8]}\t2\n")
cand = {**json.load(open(P / "bm25_candidates.json", encoding="utf-8")), **json.load(open(P / "bm25_candidates_rest.json", encoding="utf-8"))}
with open(ROOT / "data/benchmark_draft/passages.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(["passage_id", "title", "text_first_700_chars", "licence_note"])
    done = set()
    for r in bench:
        if r[8] in done: continue
        done.add(r[8])
        t = next(c for c in cand[r[0]] if c[0] == r[8])
        w.writerow([t[0], t[1], t[2], "Wikipedia CC BY-SA 4.0 (see corpus metadata)"])

st = {"items": len(rows), "by_source": Counter(r[1] for r in rows), "by_domain_youtube": Counter(r[2] for r in rows if r[1] == "youtube"),
      "by_script": Counter(r[3] for r in rows), "rerank_grade": Counter(r[6] for r in rows),
      "grade_status": Counter(r[10] for r in rows), "grade2_draft_review": Counter(r[11] for r in rows if r[6] == "2"),
      "benchmark_draft_queries": len(bench), "benchmark_dev": nd, "benchmark_test": len(bench) - nd,
      "benchmark_by_domain": Counter(r[2] or "unassigned" for r in bench), "benchmark_passages": len(done)}
(ROOT / "data/annotations/pilot/dataset_10000_stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(st, ensure_ascii=False, indent=2))
