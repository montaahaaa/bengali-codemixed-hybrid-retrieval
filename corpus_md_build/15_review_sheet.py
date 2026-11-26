import csv, json, random, collections as C
random.seed(42)
qs = [json.loads(l) for l in open("data/queries.jsonl", encoding="utf-8")]
need = C.defaultdict(dict); meta = {}
for q in qs:
    m = q["metadata"]; need[m["source_doc"]][m["script"]] = (q["_id"], q["text"]); meta[m["source_doc"]] = (m["domain"], m["split"])
full = [d for d, v in need.items() if {"bangla", "mixed", "romanized"} <= set(v)]
corpus = {}
want = set(full)
for l in open("data/corpus.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["doc_id"] in want: corpus[r["doc_id"]] = r
by = C.defaultdict(list)
for d in full: by[meta[d][0]].append(d)
quota = {"history_culture": 25, "technology": 14, "health": 14, "government_law": 14, "economy_fees": 12, "education": 14, "agriculture": 10,
         "transport_local": 10, "people": 10, "science": 8, "geography": 8, "other": 6, "sports": 0}
pick = []
for dom, k in quota.items():
    random.shuffle(by[dom]); pick += by[dom][:k]
random.shuffle(pick); pick = pick[:150]
double = set(pick[:50])                      # second reviewer rates these too
cols = ["need_id", "double_review", "domain", "passage_id", "passage_title", "passage_text",
        "q_bangla_id", "q_bangla", "q_mixed_id", "q_mixed", "q_romanized_id", "q_romanized",
        "bangla_natural_yes_no", "bangla_answerable_yes_no", "mixed_natural_yes_no", "mixed_answerable_yes_no",
        "romanized_natural_yes_no", "romanized_answerable_yes_no", "same_meaning_all_three_yes_no", "fix_or_notes", "reviewer"]
with open("data/review/review_sheet.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(cols)
    for i, d in enumerate(pick, 1):
        v = need[d]; p = corpus[d]
        w.writerow([f"R{i:03d}", "yes" if d in double else "no", meta[d][0], d, p["title"], p["text"],
                    v["bangla"][0], v["bangla"][1], v["mixed"][0], v["mixed"][1], v["romanized"][0], v["romanized"][1]] + [""] * 8 + ["" ])
print("needs", len(pick), "double", len(double), C.Counter(meta[d][0] for d in pick).most_common())
