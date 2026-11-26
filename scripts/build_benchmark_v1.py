"""Benchmark v1 draft from the filtered 10,000 set. Usage: python scripts/build_benchmark_v1.py
Inputs (data/processed): dataset_10000.csv (grades), dataset_10000_filtered.csv (notebooks/filter_queries.ipynb),
bm25_candidates*.json. Query = the original comment text (owner holds platform permission incl. public release, 2026-11).
Keeps grade-2 pairs whose draft review is keep or weak; drops 'drop'. Outputs data/benchmark/{queries.csv,qrels.tsv,passages.csv}
and counts-only stats data/annotations/pilot/benchmark_v1_stats.json. Still machine-graded: needs the pooled human judging step."""
import csv, json, random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data/processed"
rd = lambda f: list(csv.DictReader(open(P / f, encoding="utf-8-sig")))
filt = {r["id"]: r for r in rd("dataset_10000_filtered.csv")}
rows = [r for r in rd("dataset_10000.csv") if r["id"] in filt]
bench = [r for r in rows if r["rerank_grade"] == "2" and r["draft_review"] in ("keep", "weak")]
bench.sort(key=lambda r: r["id"])
random.seed(42); random.shuffle(bench)
nd = round(0.3 * len(bench))
out = ROOT / "data/benchmark"; out.mkdir(exist_ok=True, parents=True)
qid = {r["id"]: f"Q{k+1:04d}" for k, r in enumerate(bench)}
with open(out / "queries.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["qid", "split", "domain", "script", "source", "query", "query_clean_draft", "draft_review", "grade_status"])
    for k, r in enumerate(bench):
        w.writerow([qid[r["id"]], "dev" if k < nd else "test", r["domain"] or filt[r["id"]]["domain"] or "unassigned",
                    r["script"], r["source"], r["text"], r["query_clean_draft"], r["draft_review"], r["grade_status"]])
with open(out / "qrels.tsv", "w", encoding="utf-8", newline="") as f:
    for r in bench: f.write(f"{qid[r['id']]}\t0\t{r['best_passage_id']}\t2\n")
cand = {**json.load(open(P / "bm25_candidates.json", encoding="utf-8")), **json.load(open(P / "bm25_candidates_rest.json", encoding="utf-8"))}
done = set()
with open(out / "passages.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(["passage_id", "title", "text_first_700_chars", "licence_note"])
    for r in bench:
        pid = r["best_passage_id"]
        if pid in done: continue
        done.add(pid)
        t = next(c for c in cand[r["id"]] if c[0] == pid)
        w.writerow([pid, t[1], t[2], "Wikipedia CC BY-SA 4.0 (see corpus metadata)"])
st = {"queries": len(bench), "dev": nd, "test": len(bench) - nd, "passages": len(done),
      "by_domain": Counter(r["domain"] or "unassigned" for r in bench), "by_script": Counter(r["script"] for r in bench),
      "by_source": Counter(r["source"] for r in bench), "by_review": Counter(r["draft_review"] for r in bench),
      "by_grade_status": Counter(r["grade_status"] for r in bench)}
(ROOT / "data/annotations/pilot/benchmark_v1_stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(st, ensure_ascii=False, indent=2))

# flat view for reading and checking (convenience only; the three files above stay the source of truth)
pas = {r["passage_id"]: r for r in csv.DictReader(open(out / "passages.csv", encoding="utf-8-sig"))}
qs = {r["qid"]: r for r in csv.DictReader(open(out / "queries.csv", encoding="utf-8-sig"))}
with open(out / "benchmark_flat.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["qid", "split", "domain", "script", "source", "query", "query_clean_draft", "draft_review", "grade_status",
                "passage_id", "title", "passage_text", "grade", "my_check_ok_yes_no", "notes"])
    for line in open(out / "qrels.tsv", encoding="utf-8"):
        q, _, pid, g = line.rstrip("\n").split("\t")
        r, p = qs[q], pas[pid]
        w.writerow([q, r["split"], r["domain"], r["script"], r["source"], r["query"], r["query_clean_draft"], r["draft_review"],
                    r["grade_status"], pid, p["title"], p["text_first_700_chars"], g, "", ""])
