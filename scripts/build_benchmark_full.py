"""ONE benchmark: all filtered real-world queries (9,577) + MIRACL-bn queries (2,042, Apache-2.0, source = miracl_bn). Usage: python scripts/build_benchmark_full.py
Every query is kept; the tier says how much evidence it has:
  core        grade 2, draft review keep/weak     -> qrels grade 2
  partial     grade 1 (partly useful passage)     -> qrels grade 1
  unanswered  grade 0 (no passage found by BM25+reranker; may still be answerable by other systems)
  rejected    grade 2 but draft review = drop (greeting, not a query, wrong passage) -> no qrels
  synthetic   LLM-written queries from a known Bangla Wikipedia passage (split = synthetic, kept out of dev/test)
  miracl      MIRACL-bn train+dev questions with the published human relevance judgments (relevant -> grade 2)
All grades are machine-made (BM25 top 10 + bge-reranker-v2-m3) and unchecked unless grade_status = owner_confirmed.
Outputs data/benchmark_full/{queries.csv,qrels.tsv,passages.csv,benchmark_flat.csv} + counts in data/annotations/pilot/benchmark_full_stats.json"""
import csv, gzip, json, random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data/processed"
rd = lambda f: list(csv.DictReader(open(P / f, encoding="utf-8-sig")))
filt = {r["id"]: r for r in rd("dataset_10000_filtered.csv")}
rows = [r for r in rd("dataset_10000.csv") if r["id"] in filt]
cand = {**json.load(open(P / "bm25_candidates.json", encoding="utf-8")), **json.load(open(P / "bm25_candidates_rest.json", encoding="utf-8"))}

def tier(r):
    g = r["rerank_grade"]
    if g == "miracl": return "miracl"
    if g == "synthetic": return "synthetic"
    if g == "2": return "rejected" if r["draft_review"] == "drop" else "core"
    return "partial" if g == "1" else "unanswered"

RAW = ROOT / "data/raw/miracl_bn"
mq, mrel = {}, {}
for sp in ("train", "dev"):
    for l in open(RAW / f"topics.miracl-v1.0-bn-{sp}.tsv", encoding="utf-8"):
        q, t = l.rstrip("\n").split("\t", 1); mq[f"M-{sp}-{q}"] = t
    for l in open(RAW / f"qrels.miracl-v1.0-bn-{sp}.tsv", encoding="utf-8"):
        q, _, d, g = l.split()
        if g == "1": mrel.setdefault(f"M-{sp}-{q}", []).append(d)
need = {d for v in mrel.values() for d in v}
mpas = {}
with gzip.open(RAW / "docs-0.jsonl.gz", "rt", encoding="utf-8") as f:
    for l in f:
        r = json.loads(l)
        if r["docid"] in need: mpas[r["docid"]] = r
SYN = json.load(open(ROOT / "data/synthetic/synthetic_queries.json", encoding="utf-8"))
if (ROOT / "data/synthetic/synthetic_queries_api.json").exists():       # output of scripts/gen_synthetic_queries.py
    _api = json.load(open(ROOT / "data/synthetic/synthetic_queries_api.json", encoding="utf-8"))
    SYN["queries"] += _api["queries"]; SYN["passages"].update(_api["passages"])
def bn_script(t):
    n = sum("\u0980" <= c <= "\u09ff" for c in t); a = sum(c.isascii() and c.isalpha() for c in t)
    return "bn" if n + a == 0 or n / (n + a) > 0.9 else "latin" if n / (n + a) < 0.1 else "mixed"
for q, t in mq.items():     # same dict shape as the real-world rows
    rows.append({"id": q, "source": "miracl_bn", "domain": "general", "script": bn_script(t), "text": t, "query_clean_draft": "",
                 "draft_review": "", "grade_status": "published_human", "rerank_grade": "miracl", "rerank_score": "", "best_passage_id": ""})
syn_pid = {}
for k, x in enumerate(SYN["queries"], 1):
    sid = f"S-{k:04d}"; syn_pid[sid] = x["passage_id"]
    rows.append({"id": sid, "source": "llm_synthetic_" + x["style"], "domain": x["domain"], "script": bn_script(x["query"]), "text": x["query"],
                 "query_clean_draft": "", "draft_review": "", "grade_status": "llm_generated_unchecked", "rerank_grade": "synthetic",
                 "rerank_score": "", "best_passage_id": x["passage_id"]})
rows.sort(key=lambda r: r["id"])
random.seed(42); random.shuffle(rows)
real = [r for r in rows if r["rerank_grade"] != "synthetic"]
nd = round(0.3 * len(real))
split_of = {}
for k, r in enumerate(real): split_of[r["id"]] = "dev" if k < nd else "test"
for r in rows:
    if r["rerank_grade"] == "synthetic": split_of[r["id"]] = "synthetic"
out = ROOT / "data/benchmark_full"; out.mkdir(exist_ok=True, parents=True)
qid = {r["id"]: f"Q{k+1:05d}" for k, r in enumerate(rows)}
dom = lambda r: r["domain"] or (filt[r["id"]]["domain"] if r["id"] in filt else "") or "unassigned"

pas = {}
def passage(r):
    pid = r["best_passage_id"]
    if not pid: return None
    c = next((c for c in cand.get(r["id"], []) if c[0] == pid), None)
    if c: pas[pid] = (c[1], c[2])
    return pas.get(pid)

with open(out / "queries.csv", "w", encoding="utf-8-sig", newline="") as f, open(out / "qrels.tsv", "w", encoding="utf-8", newline="") as q:
    w = csv.writer(f)
    w.writerow(["qid", "split", "tier", "domain", "script", "source", "query", "query_clean_draft", "draft_review", "grade_status"])
    for k, r in enumerate(rows):
        t = tier(r)
        w.writerow([qid[r["id"]], split_of[r["id"]], t, dom(r), r["script"], r["source"], r["text"],
                    r["query_clean_draft"], r["draft_review"], r["grade_status"]])
        if t == "synthetic":
            d = syn_pid[r["id"]]; pas[d] = (SYN["passages"][d]["title"], SYN["passages"][d]["text"][:700])
            q.write(f"{qid[r['id']]}\t0\t{d}\t2\n")
        elif t == "miracl":
            for d in mrel.get(r["id"], []):
                pas[d] = (mpas[d]["title"], mpas[d]["text"][:700]); q.write(f"{qid[r['id']]}\t0\t{d}\t2\n")
        elif t in ("core", "partial") and passage(r):
            q.write(f"{qid[r['id']]}\t0\t{r['best_passage_id']}\t{r['rerank_grade']}\n")
with open(out / "passages.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(["passage_id", "title", "text_first_700_chars", "licence_note"])
    for pid, (t, x) in pas.items(): w.writerow([pid, t, x, "Wikipedia CC BY-SA 4.0 (see corpus metadata; MIRACL rows: via MIRACL, annotations Apache-2.0)"])
with open(out / "benchmark_flat.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["qid", "split", "tier", "domain", "script", "source", "query", "query_clean_draft", "draft_review", "grade_status",
                "grade", "rerank_score", "n_relevant", "passage_id", "title", "passage_text", "my_check_ok_yes_no", "notes"])
    for k, r in enumerate(rows):
        t = tier(r); p = passage(r) if t in ("core", "partial", "rejected") else None
        if t == "synthetic":
            d = syn_pid[r["id"]]; p = pas[d]; r = {**r, "best_passage_id": d, "rerank_grade": "2"}
        if t == "miracl":
            d = mrel[r["id"]][0]; p = pas[d]; r = {**r, "best_passage_id": d, "rerank_grade": "2"}
        nrel = len(mrel.get(r["id"], [])) if t == "miracl" else 1 if t in ("core", "partial", "synthetic") else 0
        w.writerow([qid[r["id"]], split_of[r["id"]], t, dom(r), r["script"], r["source"], r["text"], r["query_clean_draft"],
                    r["draft_review"], r["grade_status"], r["rerank_grade"], r["rerank_score"], nrel,
                    r["best_passage_id"] if p else "", p[0] if p else "", p[1] if p else "", "", ""])
st = {"queries": len(rows), "dev": nd, "test": len(real) - nd, "synthetic": len(rows) - len(real), "passages": len(pas), "by_tier": Counter(tier(r) for r in rows),
      "by_domain": Counter(dom(r) for r in rows), "by_script": Counter(r["script"] for r in rows),
      "by_source": Counter(r["source"] for r in rows), "by_grade_status": Counter(r["grade_status"] for r in rows),
      "tier_by_script": {s: Counter(tier(r) for r in rows if r["script"] == s) for s in ("bn", "latin", "mixed")}}
(ROOT / "data/annotations/pilot/benchmark_full_stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(st, ensure_ascii=False, indent=2))
