"""Re-grade queries against an ADDED corpus source. Usage: python scripts/regrade_new_source.py corpus_wikibooks.jsonl wikibooks
BM25 over the new source only (top 10), then bge-reranker-v2-m3 (same thresholds as rerank_grade.py: >=0.60 -> 2, >=0.20 -> 1).
Queries: Bangla/mixed rows of dataset_10000_filtered.csv whose current grade is 0 or 1. Writes data/processed/regrade_<name>.csv
with old and new grade/score and the best new passage. Private (platform text). Romanized queries are skipped: BM25 cannot reach them."""
import csv, json, sys, time
from pathlib import Path
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
sys.path.insert(0, str(Path(__file__).resolve().parent))
from find_candidates import tok

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data/processed"
src, name = sys.argv[1], sys.argv[2]
psg = [json.loads(l) for l in open(ROOT / "data/corpus" / src, encoding="utf-8")]
bm = BM25Okapi([tok(p["title"] + " " + p["text"]) for p in psg])
filt = {r["id"] for r in csv.DictReader(open(P / "dataset_10000_filtered.csv", encoding="utf-8-sig"))}
rows = [r for r in csv.DictReader(open(P / "dataset_10000.csv", encoding="utf-8-sig"))
        if r["id"] in filt and r["script"] in ("bn", "mixed") and r["rerank_grade"] in ("0", "1")]
print(len(psg), "passages;", len(rows), "queries", flush=True)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = CrossEncoder("BAAI/bge-reranker-v2-m3", device=dev, max_length=320, activation_fn=torch.nn.Sigmoid(),
                     model_kwargs={"torch_dtype": torch.float16})
t0 = time.time()
with open(P / f"regrade_{name}.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "text", "old_grade", "old_score", "bm25_top", "new_grade", "new_score", "passage_id", "title", "passage_text"])
    for k, r in enumerate(rows, 1):
        s = bm.get_scores(tok(r["text"]))
        top = sorted(range(len(s)), key=lambda i: -s[i])[:10]
        if s[top[0]] < 6:                      # almost no shared informative words: skip the slow reranker
            w.writerow([r["id"], r["text"], r["rerank_grade"], r["rerank_score"], f"{s[top[0]]:.1f}", 0, "0.000", "", "", ""]); continue
        sc = model.predict([(r["text"], f"{psg[i]['title']}. {psg[i]['text']}") for i in top], batch_size=10, show_progress_bar=False)
        b = max(range(len(top)), key=lambda j: sc[j]); p = psg[top[b]]
        g = 2 if sc[b] >= 0.60 else 1 if sc[b] >= 0.20 else 0
        w.writerow([r["id"], r["text"], r["rerank_grade"], r["rerank_score"], f"{s[top[0]]:.1f}", g, f"{sc[b]:.3f}", p["passage_id"], p["title"], p["text"][:700]])
        if k % 200 == 0: print(f"{k}/{len(rows)} {time.time()-t0:.0f}s", flush=True); f.flush()
print("DONE", flush=True)
