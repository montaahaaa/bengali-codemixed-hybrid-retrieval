import json, os, time, numpy as np, torch
from sentence_transformers import SentenceTransformer
M = "intfloat/multilingual-e5-base"
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer(M, device=dev); model.max_seq_length = 256
texts = [json.loads(l)["text"] for l in open("data/corpus.jsonl", encoding="utf-8")]
t = time.time()
emb = model.encode(["passage: " + x for x in texts], batch_size=64, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
np.save("data/emb_e5base_passages.npy", emb.astype("float16"))
print("encoded", emb.shape, "in", round(time.time() - t), "s on", dev)
