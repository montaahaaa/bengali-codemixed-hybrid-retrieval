"""Append drafted Bangla/Romanized/mixed queries to queries.jsonl and qrels.csv (mapped by 6-gram overlap with full passage text)."""
import csv, json, re, collections as C
D = "../data/annotations/pilot/draft_batch1/"
def grams(t, n=6):
    w = re.sub(r"[^\w\s]", " ", t).split(); return {" ".join(w[i:i+n]) for i in range(max(0, len(w)-n+1))}
chunks = C.defaultdict(list)
for l in open("data/corpus.jsonl", encoding="utf-8"):
    r = json.loads(l); m = re.search(r"curid=(\d+)", r["source_url"])
    if m: chunks[m.group(1)].append((r["doc_id"], grams(r["text"])))
need = {}
for f in ["all_needs_for_review.csv", "draft_needs_batch3.csv"]:
    for r in csv.DictReader(open(D+f, encoding="utf-8-sig")): need[r["need_id"]] = r
stat = C.Counter(); qrels = []; out = []
for r in csv.DictReader(open(D+"queries_long.csv", encoding="utf-8-sig")):
    n = need.get(r["need_id"]); pid = r["passage_id"]
    m = re.match(r"bn-(\d+)-\d+$", pid) or re.match(r"(\d+)#\d+$", pid)
    if not n or not m or m.group(1) not in chunks: stat["skip"] += 1; continue
    og = grams(n["passage_text"]); best = max(((len(og & g), d) for d, g in chunks[m.group(1)]), default=(0, None))
    if best[0] == 0: stat["no_overlap"] += 1; continue
    qid = "D_" + r["qid"]; stat["ok_"+r["form"]] += 1
    qrels.append((qid, best[1], 2))
    out.append({"_id": qid, "text": r["query"], "metadata": {"script": r["detected_script"], "domain": r["domain"], "source": "draft_" + r["form"], "eval_ok": "yes", "split": "draft"}})
with open("data/queries.jsonl", "a", encoding="utf-8") as f:
    for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
with open("data/qrels.csv", "a", newline="", encoding="utf-8") as f: csv.writer(f).writerows(qrels)
print(stat, len(out))
