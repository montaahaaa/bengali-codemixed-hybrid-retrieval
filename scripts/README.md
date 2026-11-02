# Scripts

- `check_setup.py` — Stage 0 smoke test (`python scripts/check_setup.py`).
- `encode_check.py` — encodes 3 sentences to confirm the encoder and device work. Plumbing only, not a result.

## Running on Kaggle (GPU)

1. New notebook, Settings → Accelerator → GPU. Turn Internet on.
2. `!git clone <repo-url> repo && cd repo`
3. `!pip install -q -r requirements.txt` (Kaggle may swap `faiss-cpu` for `faiss-gpu`; code works with either).
4. Add `HF_TOKEN` etc. under Add-ons → Secrets, not in the repo.
5. `!python scripts/encode_check.py` — expect `device=cuda`.
