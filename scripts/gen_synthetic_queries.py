"""Generate synthetic queries with the Claude API from Bangla Wikipedia passages (benchmark tier 'synthetic').
Usage: python scripts/gen_synthetic_queries.py --n 400 [--model claude-haiku-5-5] [--dry-run]
Needs ANTHROPIC_API_KEY in .env (never commit it). Resumable: results are appended to data/synthetic/api_cache.jsonl.
Steps per passage: (1) the model writes 3 queries (Bangla, Romanized, code-mixed) that the passage answers, without copying
wording; it may mark the passage unusable (list, stub, name-only page). (2) a second call checks each query: does the passage
actually answer it (yes/no). Only 'yes' queries are kept. Output data/synthetic/synthetic_queries_api.json (same schema as
synthetic_queries.json; build_benchmark_full.py merges both). Synthetic queries stay out of dev/test; all are unchecked by humans."""
import argparse, json, os, random, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/synthetic"; OUT.mkdir(exist_ok=True, parents=True)
CACHE = OUT / "api_cache.jsonl"
DOMAINS = {
 "health": r"রোগ|জ্বর|টিকা|ওষুধ|চিকিৎসা|ক্যান্সার|ডায়াবেটিস|ভিটামিন|হাসপাতাল|স্বাস্থ্য",
 "education": r"বিশ্ববিদ্যালয়|কলেজ|বিদ্যালয়|পরীক্ষা|বৃত্তি|শিক্ষা বোর্ড|ভর্তি",
 "tech": r"ইন্টারনেট|কম্পিউটার|মোবাইল|সফটওয়্যার|পাসওয়ার্ড|নেটওয়ার্ক|স্মার্টফোন|সাইবার",
 "fees": r"ব্যাংক|কর\b|আয়কর|বীমা|ঋণ|সুদ|টাকা|শেয়ার|বাজেট",
 "procedures": r"পাসপোর্ট|নিবন্ধন|জাতীয় পরিচয়|ভিসা|নাগরিকত্ব|আদালত|লাইসেন্স|আইন",
 "local_services": r"রেল|বিমানবন্দর|বিদ্যুৎ|বাস|সড়ক|ডাক|মেট্রো|জরুরি",
}
GEN = """You write search queries for a Bangla retrieval benchmark. Read the passage and write 3 questions a real person in Bangladesh could type into a search box, each answerable ONLY from this passage:
1. "bangla": Bangla script.
2. "romanized": the same kind of question in Romanized Bangla (Latin letters, informal spelling like real chat messages).
3. "code_mixed": Bangla with some English words mixed in.
Rules: do not copy more than two consecutive words from the passage; paraphrase; do not use the article title as the whole query; no yes/no questions that can be guessed; the three queries should ask about different facts when possible. If the passage is a bare list, stub, disambiguation, or only a name/date with no usable fact, return {"usable": false}.
Return ONLY JSON: {"usable": true, "queries": {"bangla": "...", "romanized": "...", "code_mixed": "..."}}

Title: %s
Passage: %s"""
CHK = """Passage: %s

Question: %s

Does the passage contain the answer to the question? Reply with exactly one word: yes or no."""


def key():
    k = os.environ.get("ANTHROPIC_API_KEY", "")
    env = ROOT / ".env"
    if not k and env.exists():
        for l in env.read_text().splitlines():
            if l.startswith("ANTHROPIC_API_KEY="): k = l.split("=", 1)[1].strip().strip('"')
    return k


def pick(n, seed):
    taken = {json.loads(l)["passage_id"] for l in open(ROOT / "data/synthetic/synthetic_queries.json", encoding="utf-8") if False} if False else set()
    old = json.load(open(OUT / "synthetic_queries.json", encoding="utf-8"))["passages"].keys()
    rx = {d: re.compile(p) for d, p in DOMAINS.items()}
    pool = {d: [] for d in DOMAINS}
    for l in open(ROOT / "data/corpus/corpus_bn.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if not r["passage_id"].endswith("-0") or r["passage_id"] in old: continue
        if not 70 <= len(r["text"].split()) <= 150: continue
        for d, x in rx.items():
            if len(x.findall(r["text"])) >= 2: pool[d].append(r); break
    random.seed(seed); sel = []
    per = n // len(DOMAINS) + 1
    for d in DOMAINS:
        random.shuffle(pool[d]); sel += [(d, r) for r in pool[d][:per]]
    random.shuffle(sel)
    return sel[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400); ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--seed", type=int, default=11); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sel = pick(a.n, a.seed)
    print(len(sel), "passages;", "about", len(sel) * 4, "API calls (1 generate + 3 checks each)")
    if a.dry_run: return
    k = key()
    if not k: sys.exit("ANTHROPIC_API_KEY missing. Add it to .env (see .env.example).")
    import anthropic
    cl = anthropic.Anthropic(api_key=k)
    done = {json.loads(l)["passage_id"] for l in CACHE.open(encoding="utf-8")} if CACHE.exists() else set()
    def ask(p, mt):      # some models return a thinking block first: keep only the text blocks
        r = cl.messages.create(model=a.model, max_tokens=mt, messages=[{"role": "user", "content": p}])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()
    with CACHE.open("a", encoding="utf-8") as f:
        for i, (d, r) in enumerate(sel, 1):
            if r["passage_id"] in done: continue
            try:
                txt = ask(GEN % (r["title"], r["text"]), 3000)
                j = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
                kept = {}
                if j.get("usable"):
                    for style, q in j["queries"].items():
                        if ask(CHK % (r["text"], q), 1000).lower().startswith("yes"): kept[style] = q
                f.write(json.dumps({"passage_id": r["passage_id"], "domain": d, "title": r["title"], "text": r["text"], "kept": kept}, ensure_ascii=False) + "\n"); f.flush()
            except Exception as e:
                print("skip", r["passage_id"], str(e)[:80]); time.sleep(2)
            if i % 25 == 0: print(i, "/", len(sel), flush=True)
    out_q, out_p = [], {}
    for l in CACHE.open(encoding="utf-8"):
        c = json.loads(l)
        for style, q in c["kept"].items():
            out_q.append({"passage_id": c["passage_id"], "domain": c["domain"], "style": style, "query": q})
            out_p[c["passage_id"]] = {"title": c["title"], "text": c["text"]}
    json.dump({"generator": f"{a.model} via API, with a yes/no answerability check", "status": "unchecked", "queries": out_q, "passages": out_p},
              open(OUT / "synthetic_queries_api.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("kept queries", len(out_q), "from", len(out_p), "passages")


if __name__ == "__main__":
    main()
