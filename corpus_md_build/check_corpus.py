import json
from collections import Counter

corpus_path = "data/corpus.jsonl"

passages = []

with open(corpus_path, encoding="utf-8") as file:
    for line in file:
        passages.append(json.loads(line))

doc_ids = [p["doc_id"] for p in passages]
texts = [p["text"] for p in passages]
domains = [p["domain"] for p in passages]

print("Total passages:", len(passages))
print("Unique doc IDs:", len(set(doc_ids)))
print("Unique passage texts:", len(set(texts)))
print("Duplicate doc IDs:", len(doc_ids) - len(set(doc_ids)))
print("Duplicate texts:", len(texts) - len(set(texts)))
print("Average words per passage:", sum(len(t.split()) for t in texts) / len(passages))
print("Domain distribution:")

for domain, count in Counter(domains).most_common():
    print(f"- {domain}: {count}")
