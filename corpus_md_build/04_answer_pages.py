"""Collect Wikipedia page IDs that are answer passages (grade>=1) for benchmark queries and drafted needs,
then report how many exist in the downloaded Wikipedia file."""
import csv, json, re, collections as C
R = "../data/"
q = {r["qid"]: r for r in csv.DictReader(open(R+"benchmark_full/queries.csv", encoding="utf-8-sig"))}
pages = C.defaultdict(set)          # page id -> set of qids / need ids
src = C.Counter()
for l in open(R+"benchmark_full/qrels.tsv", encoding="utf-8"):
    qid, _, pid, g = l.rstrip("\n").split("\t")
    if int(g) < 1 or qid not in q: continue
    m = re.match(r"bn-(\d+)-\d+$", pid) or re.match(r"(\d+)#\d+$", pid)
    if m: pages[m.group(1)].add(qid); src["qrels"] += 1
    else: src["skipped:"+pid.split("-")[0]] += 1
for f in ["all_needs_for_review.csv", "draft_needs_batch3.csv"]:
    for r in csv.DictReader(open(R+"annotations/pilot/draft_batch1/"+f, encoding="utf-8-sig")):
        pid = r["passage_id"]; m = re.match(r"bn-(\d+)-\d+$", pid) or re.match(r"(\d+)#\d+$", pid)
        if m: pages[m.group(1)].add(r["need_id"]); src["drafts"] += 1
        else: src["skipped_draft:"+re.sub(r"\d+", "N", pid)] += 1
have = set()
for l in open("data/raw/wikipedia.jsonl", encoding="utf-8"):
    m = re.search(r"curid=(\d+)", json.loads(l)["url"]); 
    if m: have.add(m.group(1))
found = [p for p in pages if p in have]
print(src); print("answer pages wanted:", len(pages), "| present in wikipedia.jsonl:", len(found))
json.dump(sorted(found), open("data/answer_pages.json", "w"))
eok = {x for x in q if q[x]["eval_ok"] == "yes"}
cov = {x for p in found for x in pages[p] if x in eok}
print("eval_ok queries with answer page present:", len(cov), "of", len(eok))
