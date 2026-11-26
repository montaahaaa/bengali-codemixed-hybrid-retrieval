"""Parallel benchmark v2, Groq generator. Same task and prompts as 21_make_parallel.py (reads GEN and CHK from it), but only for
source passages that are NOT yet in data/parallel_cache.jsonl, and writes to its own file data/parallel_cache_groq.jsonl
(each row has "generator"), so the Claude-made rows are never touched. Resumable.
Usage: python 21b_make_parallel_groq.py --limit 30        (pilot on 30 new passages)
       python 21b_make_parallel_groq.py --limit 0         (all remaining)
Needs GROQ_API_KEY in ../.env."""
import argparse, json, os, re, sys, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="openai/gpt-oss-120b"); ap.add_argument("--limit", type=int, default=30); ap.add_argument("--workers", type=int, default=4)
a = ap.parse_args()
key = os.environ.get("GROQ_API_KEY", "")
if not key:
    for l in open("../.env").read().splitlines():
        if l.startswith("GROQ_API_KEY="): key = l.split("=", 1)[1].strip().strip('"')
if not key: sys.exit("GROQ_API_KEY missing in ../.env")
H = {"Authorization": "Bearer " + key}; BASE = "https://api.groq.com/openai/v1"
src = open("21_make_parallel.py", encoding="utf-8").read()
GEN = src.split('GEN = """')[1].split('"""')[0]; CHK = src.split('CHK = """')[1].split('"""')[0]
bn = lambda s: len(re.findall(r"[ঀ-৿]", s)); la = lambda s: len(re.findall(r"[A-Za-z]", s))
ok_script = {"code_mixed": lambda s: bn(s) > 0 and la(s) > 0, "romanized": lambda s: bn(s) == 0 and la(s) > 0}
base = {}
for l in open("data/queries.jsonl", encoding="utf-8"):
    q = json.loads(l); m = q["metadata"]
    if m["script"] == "bangla": base[m["source_doc"]] = (q["_id"], q["text"], m["domain"], m["split"])
OLD = "data/parallel_cache.jsonl"; NEW = "data/parallel_cache_groq.jsonl"
done = set()
for p in (OLD, NEW):
    if os.path.exists(p): done |= {json.loads(l)["doc_id"] for l in open(p, encoding="utf-8")}
items = [it for it in base.items() if it[0] not in done]
print("remaining passages:", len(items), "| already done:", len(done))
if a.limit: items = items[:a.limit]
lock = threading.Lock(); stats = {"docs": 0, "both_kept": 0, "fail": 0}
def ask(p, mt):
    body = json.dumps({"model": a.model, "max_tokens": max(mt, 2500), "temperature": 0.2,
        **({"reasoning_effort": "low"} if "gpt-oss" in a.model else {}), "messages": [{"role": "user", "content": p}]}).encode()
    for t in range(8):
        req = urllib.request.Request(BASE + "/chat/completions", data=body, headers={**H, "Content-Type": "application/json", "User-Agent": "bench-gen/1.0"})
        try:
            d = json.load(urllib.request.urlopen(req, timeout=90))
        except urllib.error.HTTPError as e:
            if e.code == 429: time.sleep(5 * (t + 1)); continue
            raise
        return re.sub(r"<think>.*?</think>", "", d["choices"][0]["message"]["content"] or "", flags=re.S).strip()
    raise RuntimeError("rate limited")
def work(it):
    doc, (qid, b, dom, split) = it
    try:
        j = json.loads(re.search(r"\{.*\}", ask(GEN % b, 1500), re.S).group(0)); kept = {}
        for st in ("code_mixed", "romanized"):
            v = (j.get(st) or "").strip()
            if v and ok_script[st](v) and ask(CHK % (b, v), 20).lower().startswith("yes"): kept[st] = v
        with lock, open(NEW, "a", encoding="utf-8") as f:
            f.write(json.dumps({"doc_id": doc, "bangla": b, "bangla_id": qid, "domain": dom, "split": split, "gen": j, "kept": kept, "generator": a.model}, ensure_ascii=False) + "\n")
        with lock: stats["docs"] += 1; stats["both_kept"] += len(kept) == 2
    except Exception as e:
        with lock: stats["fail"] += 1
        print("skip", doc, str(e)[:80])
t = time.time()
with ThreadPoolExecutor(a.workers) as ex: list(ex.map(work, items))
print(stats, "in", round(time.time() - t), "s")
