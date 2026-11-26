"""Step 4: review sheet. 34 parallel needs (102 queries), random with fixed seed, spread over domains.
Reviewer A gets all 34 needs; reviewer B gets the first 17 (51 queries) = the double-reviewed part."""
import csv, json, random, collections as C
random.seed(2026)
corpus = {}
for l in open("data/corpus.jsonl", encoding="utf-8"):
    r = json.loads(l); corpus[r["doc_id"]] = r["text"]
qs = [json.loads(l) for l in open("data/v2/queries.jsonl", encoding="utf-8")]
by = C.defaultdict(dict)
for q in qs: by[q["metadata"]["need"]][q["metadata"]["script"]] = q
needs = [n for n, d in by.items() if len(d) == 3 and by[n]["bangla"]["metadata"]["source_doc"] in corpus]
dom = C.defaultdict(list)
for n in needs: dom[by[n]["bangla"]["metadata"]["domain"]].append(n)
pick = []; pools = {d: random.sample(v, len(v)) for d, v in dom.items()}
while len(pick) < 34:                       # round-robin over domains
    for d in sorted(pools):
        if pools[d] and len(pick) < 34: pick.append(pools[d].pop())
random.shuffle(pick)
H = ["need_id", "query_id", "script", "query", "natural_yes_no", "answerable_from_passage_yes_no", "same_meaning_as_bangla_yes_no", "corrected_query_if_needed", "note", "passage"]
def write(path, ns):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(H)
        for n in ns:
            for sc in ("bangla", "mixed", "romanized"):
                q = by[n][sc]; w.writerow([n, q["_id"], sc, q["text"], "", "", "" if sc != "bangla" else "(reference)", "", "", corpus[q["metadata"]["source_doc"]]])
write("../review/review_sheet_A.csv", pick); write("../review/review_sheet_B.csv", pick[:17])
print(len(pick), C.Counter(by[n]["bangla"]["metadata"]["domain"] for n in pick))
