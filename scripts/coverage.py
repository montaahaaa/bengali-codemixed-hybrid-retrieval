"""Coverage rate from the hand-filled questions CSV -> stage1_coverage.json."""
import csv, json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/annotations/pilot/stage1_questions.csv"
corpus_ids = {json.loads(l)["passage_id"] for l in open(ROOT / "data/corpus/corpus.jsonl", encoding="utf-8")}
rows = [r for r in csv.DictReader(open(SRC, encoding="utf-8-sig")) if r["question"].strip()]
assert rows, "no questions written yet"
bad = [r["qid"] for r in rows if r["has_answer"] not in ("yes", "no", "partial")]
assert not bad, f"has_answer must be yes/no/partial: {bad}"
missing = [r["qid"] for r in rows if r["has_answer"] == "yes" and r["answer_passage_id"] not in corpus_ids]
assert not missing, f"answer_passage_id not in corpus: {missing}"
out = {"questions": len(rows), "by_answer": dict(Counter(r["has_answer"] for r in rows)), "per_domain": {}}
for r in rows:
    d = out["per_domain"].setdefault(r["domain"], {"n": 0, "yes": 0, "partial": 0, "no": 0})
    d["n"] += 1
    d[r["has_answer"]] += 1
out["coverage_yes"] = round(out["by_answer"].get("yes", 0) / len(rows), 3)
out["coverage_yes_or_partial"] = round((out["by_answer"].get("yes", 0) + out["by_answer"].get("partial", 0)) / len(rows), 3)
(ROOT / "data/annotations/pilot/stage1_coverage.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
