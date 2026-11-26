"""Download Wikipedia parquet shards (wikimedia/wikipedia) into data/raw/wikipedia/.

Usage: python scripts/download_wikipedia.py bn --snapshot 20231101
English is big (~11 GB), so use filter_english.py, which downloads one shard
at a time, filters it and keeps the shard on disk.
"""
import argparse, os, time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # the xet transfer path failed mid-run
from huggingface_hub import HfApi, hf_hub_download

REPO = "wikimedia/wikipedia"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/raw/wikipedia"


def shards(lang, snapshot):
    prefix = f"{snapshot}.{lang}/"
    return sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
                  if f.startswith(prefix) and f.endswith(".parquet"))


def fetch(path, tries=6):
    """Download one shard (skipped if already on disk). Retries on network errors."""
    for i in range(tries):
        try:
            return hf_hub_download(REPO, path, repo_type="dataset", local_dir=OUT)
        except Exception as e:
            print(f"download failed ({i + 1}/{tries}): {str(e)[:120]}", flush=True)
            time.sleep(min(60, 5 * 2 ** i))
    raise RuntimeError(f"could not download {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("lang", choices=["bn"])
    ap.add_argument("--snapshot", required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for s in shards(a.lang, a.snapshot):
        print("fetching", s)
        fetch(s)
