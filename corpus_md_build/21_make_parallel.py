"""Parallel benchmark v2: for each source passage keep the Bangla query of v1 and ask Claude for a code-mixed and a Romanized version
of the SAME question (same fact, same answer). A second call checks equivalence (yes/no); only 'yes' versions are kept.
Usage: python 21_make_parallel.py --n 12 (pilot) | --n 0 (all). Resumable via data/parallel_cache.jsonl."""
import argparse, json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import anthropic
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=12); ap.add_argument("--model", default="claude-haiku-5-5"); ap.add_argument("--workers", type=int, default=16)
a = ap.parse_args()
GEN = """Below is a question in Bangla script. Write two other versions of EXACTLY the same question, asking for exactly the same fact (the answer must be identical):
1. "code_mixed": Bangla grammar and mostly Bangla script, with some English words in Latin script the way Bangladeshis naturally mix them (for example technical or place terms).
2. "romanized": the same question written entirely in Latin letters (Romanized Bangla), informal chat spelling, English words kept as English.
Do not add, drop or change any detail (names, years, numbers, what is asked). Do not answer the question.
Return ONLY JSON: {"code_mixed": "...", "romanized": "..."}

Question: %s"""
CHK = """Question A (Bangla): %s
Question B: %s

Do A and B ask for exactly the same fact, so that the same answer would be correct for both? Reply with exactly one word: yes or no."""
k = os.environ.get("ANTHROPIC_API_KEY", "")
if not k:
    for l in open("../.env").read().splitlines():
        if l.startswith("ANTHROPIC_API_KEY="): k = l.split("=", 1)[1].strip().strip('"')
cl = anthropic.Anthropic(api_key=k)
base = {}
for l in open("data/queries.jsonl", encoding="utf-8"):
    q = json.loads(l); m = q["metadata"]
    if m["script"] == "bangla": base[m["source_doc"]] = (q["_id"], q["text"], m["domain"], m["split"])
items = list(base.items())[:a.n] if a.n else list(base.items())
CACHE = "data/parallel_cache.jsonl"
done = {json.loads(l)["doc_id"] for l in open(CACHE)} if os.path.exists(CACHE) else set()
lock = threading.Lock()
def ask(p, mt):
    r = cl.messages.create(model=a.model, max_tokens=mt, messages=[{"role": "user", "content": p}])
    return "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()
def work(it):
    doc, (qid, bn, dom, split) = it
    if doc in done: return
    try:
        j = json.loads(re.search(r"\{.*\}", ask(GEN % bn, 1500), re.S).group(0)); kept = {}
        for st in ("code_mixed", "romanized"):
            if j.get(st) and ask(CHK % (bn, j[st]), 500).lower().startswith("yes"): kept[st] = j[st].strip()
        with lock, open(CACHE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"doc_id": doc, "bangla": bn, "bangla_id": qid, "domain": dom, "split": split, "gen": j, "kept": kept}, ensure_ascii=False) + "\n")
    except Exception as e: print("skip", doc, str(e)[:80])
t = time.time()
with ThreadPoolExecutor(a.workers) as ex: list(ex.map(work, items))
print("done", len(items), "in", round(time.time() - t), "s")
