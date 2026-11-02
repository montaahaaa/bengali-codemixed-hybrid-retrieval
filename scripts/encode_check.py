"""Plumbing check: encode three short sentences. Not a benchmark result."""
import random, time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SENTENCES = [
    "পাসপোর্ট নবায়ন করতে কত টাকা লাগে?",
    "How much does passport renewal cost?",
    "passport renew korte koto taka lagbe?",
]


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def encode(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = pick_device()
    model = SentenceTransformer(MODEL, device=device)
    t = time.time()
    emb = model.encode(SENTENCES)
    return emb.shape, device, time.time() - t


if __name__ == "__main__":
    shape, device, secs = encode()
    print(f"shape={shape} device={device} seconds={secs:.2f}")
