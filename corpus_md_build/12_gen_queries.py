"""Passage-first query generation from data/corpus.jsonl with the Claude API (+ yes/no answer check per query).
Usage: python 12_gen_queries.py --n 30 [--model claude-haiku-5-5] [--workers 8]. Resumable via data/gen_cache.jsonl.
Queries are LLM-written and unchecked by humans. Only public corpus passages are sent."""
import argparse, json, os, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "../scripts")
DOMAINS = {
 "health": r"রোগ|জ্বর|টিকা|ওষুধ|চিকিৎসা|ক্যান্সার|ডায়াবেটিস|ভিটামিন|হাসপাতাল|স্বাস্থ্য",
 "education": r"বিশ্ববিদ্যালয়|কলেজ|বিদ্যালয়|পরীক্ষা|বৃত্তি|শিক্ষা বোর্ড|ভর্তি",
 "technology": r"ইন্টারনেট|কম্পিউটার|মোবাইল|সফটওয়্যার|পাসওয়ার্ড|নেটওয়ার্ক|স্মার্টফোন|সাইবার",
 "economy_fees": r"ব্যাংক|আয়কর|বীমা|ঋণ|সুদ|মুদ্রা|শেয়ার|বাজেট|ভ্যাট",
 "government_law": r"পাসপোর্ট|নিবন্ধন|জাতীয় পরিচয়|ভিসা|নাগরিকত্ব|আদালত|লাইসেন্স|আইন|সরকার",
 "agriculture": r"কৃষি|ধান|ফসল|সার\b|বীজ|সেচ|মাছ চাষ|পশুপালন",
 "transport_local": r"রেল|বিমানবন্দর|বিদ্যুৎ|বাস\b|সড়ক|ডাক|মেট্রো|জরুরি",
 "history_culture": r"ইতিহাস|যুদ্ধ|সংস্কৃতি|উৎসব|সাহিত্য|কবি|সঙ্গীত|ভাষা আন্দোলন",
}
GEN = """You write search queries for a Bangla retrieval benchmark. Read the passage and write 3 questions a real person in Bangladesh could type into a search box, each answerable ONLY from this passage:
1. "bangla": Bangla script.
2. "romanized": the same kind of question in Romanized Bangla (Latin letters, informal spelling like real chat messages).
3. "code_mixed": Bangla script with some English words mixed in, as Bangladeshis really write.
Rules: do not copy more than two consecutive words from the passage; paraphrase; do not use the article title as the whole query; no yes/no questions that can be guessed; the three queries should ask about different facts when possible. If the passage is a bare list, stub, disambiguation, or only a name/date with no usable fact, return {"usable": false}.
Also give the topic area of the passage as "domain", one of: health, education, technology, economy_fees, government_law, agriculture, transport_local, history_culture, science, geography, people, other.
Return ONLY JSON: {"usable": true, "domain": "...", "queries": {"bangla": "...", "romanized": "...", "code_mixed": "..."}}

Title: %s
Passage: %s"""
CHK = """Passage: %s

Question: %s

Does the passage contain the answer to the question? Reply with exactly one word: yes or no."""
def key():
    k = os.environ.get("ANTHROPIC_API_KEY", "")
    if not k:
        for l in open("../.env").read().splitlines():
            if l.startswith("ANTHROPIC_API_KEY="): k = l.split("=", 1)[1].strip().strip('"')
    return k
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30); ap.add_argument("--model", default="claude-haiku-5-5")
ap.add_argument("--seed", type=int, default=11); ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
rx = {d: re.compile(p) for d, p in DOMAINS.items()}; pool = {d: [] for d in DOMAINS}
for l in open("data/corpus.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if not 70 <= len(r["text"].split()) <= 160 or r["chunk_id"] != 0 and random.random() < 0.7: continue
    for d, x in rx.items():
        if len(x.findall(r["text"])) >= 2: pool[d].append(r); break
random.seed(a.seed); sel = []
for d in DOMAINS: random.shuffle(pool[d])
cap = a.n // len(DOMAINS)
for d in DOMAINS: sel += [(d, r) for r in pool[d][:cap]]
left = a.n - len(sel); extra = [(d, r) for d in DOMAINS for r in pool[d][cap:]]; random.shuffle(extra); sel += extra[:left]
random.shuffle(sel)
print({d: len(v) for d, v in pool.items()}, "| selected", len(sel))
import anthropic
cl = anthropic.Anthropic(api_key=key())
CACHE = "data/gen_cache.jsonl"
done = {json.loads(l)["doc_id"] for l in open(CACHE)} if os.path.exists(CACHE) else set()
lock = threading.Lock()
def ask(p, mt):
    r = cl.messages.create(model=a.model, max_tokens=mt, messages=[{"role": "user", "content": p}])
    return "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()
def work(item):
    d, r = item
    if r["doc_id"] in done: return
    try:
        j = json.loads(re.search(r"\{.*\}", ask(GEN % (r["title"], r["text"]), 3000), re.S).group(0))
        kept = {}
        if j.get("usable"):
            for st, q in j["queries"].items():
                if ask(CHK % (r["text"], q), 1000).lower().startswith("yes"): kept[st] = q
        with lock, open(CACHE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"doc_id": r["doc_id"], "domain": j.get("domain") or d, "kw_domain": d, "title": r["title"], "usable": bool(j.get("usable")), "kept": kept}, ensure_ascii=False) + "\n")
    except Exception as e: print("skip", r["doc_id"], str(e)[:80])
t = time.time()
with ThreadPoolExecutor(a.workers) as ex: list(ex.map(work, sel))
print("done in", round(time.time() - t), "s")
