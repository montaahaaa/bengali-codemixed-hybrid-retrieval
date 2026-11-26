"""Write queries.jsonl and qrels.csv (BEIR style) for the rebuilt corpus.
Old answer passages are mapped to new chunks of the same Wikipedia page by word-8gram overlap."""
import csv, json, re, collections as C
R = "../data/benchmark_full/"
def grams(t, n=6):
    w = re.sub(r"[^\w\s]", " ", t).split(); return {" ".join(w[i:i+n]) for i in range(max(0, len(w)-n+1))}
page_chunks = C.defaultdict(list)
for l in open("data/corpus.jsonl", encoding="utf-8"):
    r = json.loads(l); m = re.search(r"curid=(\d+)", r["source_url"])
    if m: page_chunks[m.group(1)].append((r["doc_id"], grams(r["text"]), r["text"]))
old = {r["passage_id"]: r["text_first_700_chars"] for r in csv.DictReader(open(R+"passages.csv", encoding="utf-8-sig"))}
q = {r["qid"]: r for r in csv.DictReader(open(R+"queries.csv", encoding="utf-8-sig"))}
split = {r["qid"]: r for r in csv.DictReader(open(R+"splits_v2.csv", encoding="utf-8-sig"))}
rows, stat = [], C.Counter()
for l in open(R+"qrels.tsv", encoding="utf-8"):
    qid, _, pid, g = l.rstrip("\n").split("\t"); g = int(g)
    if g < 1 or qid not in q: continue
    m = re.match(r"bn-(\d+)-\d+$", pid) or re.match(r"(\d+)#\d+$", pid)
    if not m or m.group(1) not in page_chunks: stat["no_page"] += 1; continue
    og = grams(old.get(pid, ""))
    best = sorted(((len(og & cg), did) for did, cg, _ in page_chunks[m.group(1)]), reverse=True)
    if best and best[0][0] > 0 and og:
        rows.append((qid, best[0][1], g)); stat["mapped"] += 1
    else:
        stat["page_present_no_overlap"] += 1
print(stat)
keep = {r[0] for r in rows}
with open("data/qrels.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["query-id", "corpus-id", "score"]); w.writerows(sorted(set(rows)))
n = 0
with open("data/queries.jsonl", "w", encoding="utf-8") as f:
    for qid in sorted(keep):
        r = q[qid]
        f.write(json.dumps({"_id": qid, "text": r["query"], "metadata": {"script": r["script"], "domain": r["domain"], "source": r["source"], "eval_ok": r["eval_ok"], "split": split.get(qid, {}).get("split", r["split"])}}, ensure_ascii=False) + "\n"); n += 1
print("queries.jsonl:", n, "| qrels rows:", len(set(rows)))
ok = [x for x in keep if q[x]["eval_ok"] == "yes"]; print("eval_ok with qrels:", len(ok), C.Counter(q[x]["script"] for x in ok))
