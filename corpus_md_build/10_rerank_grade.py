"""Grade BM25 top-K candidates of each unique new query with bge-reranker-v2-m3. Writes data/rerank_scores.jsonl
(one line per query: all candidate scores). Thresholds (>=0.60 grade 2, >=0.20 grade 1) are uncalibrated."""
import json, sys, time, os
import torch
from sentence_transformers import CrossEncoder
K = 10; LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None
OUT = "data/rerank_scores.jsonl"
corpus = {}
for l in open("data/corpus.jsonl", encoding="utf-8"):
    d = json.loads(l); corpus[d["doc_id"]] = d["title"] + ". " + d["text"]
q = {}
for l in open("data/queries.jsonl", encoding="utf-8"):
    r = json.loads(l); q[r["_id"]] = r["text"]
cands = [json.loads(l) for l in open("data/candidates_bm25.jsonl", encoding="utf-8")]
done = set()
if os.path.exists(OUT): done = {json.loads(l)["qid"] for l in open(OUT)}
cands = [c for c in cands if c["qid"] not in done][:LIMIT]
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = CrossEncoder("BAAI/bge-reranker-v2-m3", device=dev, max_length=320, activation_fn=torch.nn.Sigmoid(), model_kwargs={"torch_dtype": torch.float16})
t0 = time.time()
with open(OUT, "a", encoding="utf-8") as f:
    for k, c in enumerate(cands, 1):
        top = [d for d, _ in c["cands"][:K]]
        sc = model.predict([(q[c["qid"]], corpus[d]) for d in top], batch_size=10, show_progress_bar=False)
        f.write(json.dumps({"qid": c["qid"], "scores": {d: round(float(s), 4) for d, s in zip(top, sc)}}) + "\n")
        if k % 100 == 0: f.flush(); print(f"{k}/{len(cands)} {time.time()-t0:.0f}s", flush=True)
print("done", len(cands), round(time.time() - t0), "s")
