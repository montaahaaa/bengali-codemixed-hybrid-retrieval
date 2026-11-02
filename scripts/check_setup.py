"""Stage 0 smoke test. Exit 0 if setup is complete, else non-zero."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOLDERS = ["data/raw", "data/processed", "data/corpus", "data/annotations",
           "models", "experiments", "paper", "scripts"]
HEADER = "run_name,date,config,metric,value"
SECRET_HINTS = ("hf_", "sk-")


def check_python():
    return [] if sys.version_info[:2] == (3, 12) else [f"Python is {sys.version_info[:2]}, need 3.12"]


def check_folders():
    return [f"missing folder: {f}" for f in FOLDERS if not (ROOT / f).is_dir()]


def check_tracker():
    p = ROOT / "experiments/experiment_tracker.csv"
    if not p.exists():
        return ["missing experiments/experiment_tracker.csv"]
    first = p.read_text().splitlines()[0] if p.read_text() else ""
    return [] if first == HEADER else [f"tracker header is {first!r}, need {HEADER!r}"]


def check_env_example():
    p = ROOT / ".env.example"
    if not p.exists():
        return ["missing .env.example"]
    bad = [l for l in p.read_text().splitlines()
           if "=" in l and not l.startswith("#") and any(h in l.split("=", 1)[1] for h in SECRET_HINTS)]
    return [f"possible secret in .env.example: {l}" for l in bad]


def check_imports():
    errs = []
    for m in ["sentence_transformers", "faiss", "rank_bm25", "pytrec_eval", "datasets", "dotenv", "numpy"]:
        try:
            __import__(m)
        except Exception as e:
            errs.append(f"import {m} failed: {e}")
    return errs


def check_encode():
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from encode_check import encode, SENTENCES
        shape, device, _ = encode()
    except Exception as e:
        return [f"test encode failed: {e}"]
    return [] if shape[0] == len(SENTENCES) and shape[1] > 0 else [f"unexpected shape {shape}"]


FAST = [check_python, check_folders, check_tracker, check_env_example]
SLOW = [check_imports, check_encode]

if __name__ == "__main__":
    failed = []
    for c in FAST + SLOW:
        errs = c()
        print(("FAIL " if errs else "ok   ") + c.__name__)
        failed += errs
    for e in failed:
        print("  -", e)
    sys.exit(1 if failed else 0)
