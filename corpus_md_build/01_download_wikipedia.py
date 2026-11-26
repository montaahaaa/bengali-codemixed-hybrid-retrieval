from datasets import load_dataset
import json
import os

OUT_PATH = "data/raw/wikipedia.jsonl"
DATASET_NAME = "sagorsarker/bangla-wikipedia"
CONFIG_NAME = "20250620"

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

print("Downloading Bengali Wikipedia dataset...")
dataset = load_dataset(DATASET_NAME, CONFIG_NAME, split="train")

count = 0

with open(OUT_PATH, "w", encoding="utf-8") as out_file:
    for row in dataset:
        title = str(row.get("title", "")).strip()
        text = str(row.get("content", row.get("text", ""))).strip()

        if not title or len(text.split()) < 100:
            continue

        record = {
            "source": "bengali_wikipedia",
            "source_id": str(row.get("id", "")),
            "title": title,
            "url": str(row.get("url", "")),
            "text": text,
            "language": "bn"
        }

        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1

print(f"Saved {count} Bengali Wikipedia articles to {OUT_PATH}")
