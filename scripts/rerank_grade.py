"""Second grading pass (phase 2.3 helper): rerank the BM25 top-10 of every query with a multilingual cross-encoder
(BAAI/bge-reranker-v2-m3) and grade from the best passage score. Usage: python scripts/rerank_grade.py
Needs data/processed/graded_queries.csv and bm25_candidates.json (from grade_queries.py). Writes
data/processed/graded_queries_rerank.csv (git-ignored, private text).
Grade from sigmoid score of the best passage: >=0.60 -> 2, >=0.20 -> 1, else 0. Thresholds are UNCALIBRATED
guesses; recalibrate on your manual grades. The reranker only reorders BM25's candidates, so it cannot recover
queries whose answer BM25 never retrieved (mostly Romanized Bangla)."""
import csv, json, sys, time
from pathlib import Path
import torch
from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parent.parent
SUF = "_rest" if "--rest" in sys.argv else ""
rows = list(csv.DictReader(open(ROOT / f"data/processed/graded_queries{SUF}.csv", encoding="utf-8-sig")))
cands = json.load(open(ROOT / f"data/processed/bm25_candidates{SUF}.json", encoding="utf-8"))
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = CrossEncoder("BAAI/bge-reranker-v2-m3", device=dev, max_length=320,
                     activation_fn=torch.nn.Sigmoid(), model_kwargs={"torch_dtype": torch.float16})
t0 = time.time()
cols = ["id", "source", "domain", "script", "text", "auto_grade", "rerank_grade", "rerank_score",
        "best_id", "best_title", "best_text", "second_score", "manual_grade", "notes"]
with (ROOT / f"data/processed/graded_queries{SUF}_rerank.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(cols)
    for k, r in enumerate(rows, 1):
        c = cands.get(r["id"], [])
        if not c:
            w.writerow([r[x] for x in cols[:6]] + [0, "0.000", "", "", "", "0.000", "", ""]); continue
        sc = model.predict([(r["text"], f"{t}. {x}") for _, t, x in c], batch_size=10, show_progress_bar=False)
        order = sorted(range(len(c)), key=lambda i: -sc[i])
        b = order[0]; g = 2 if sc[b] >= 0.60 else 1 if sc[b] >= 0.20 else 0
        w.writerow([r[x] for x in cols[:6]] + [g, f"{sc[b]:.3f}", c[b][0], c[b][1], c[b][2][:500],
                    f"{sc[order[1]]:.3f}" if len(order) > 1 else "0.000", "", ""])
        if k % 200 == 0:
            print(f"{k}/{len(rows)} {time.time()-t0:.0f}s", flush=True); f.flush()
print("done")
