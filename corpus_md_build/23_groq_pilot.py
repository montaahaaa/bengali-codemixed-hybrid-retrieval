"""Groq pilot: same task as 21_make_parallel.py on passages Haiku already did, so the two can be compared side by side.
Usage: python 23_groq_pilot.py [--model MODEL] [--n 30]. Needs GROQ_API_KEY in ../.env. Writes data/groq_pilot.jsonl and prints pass rates."""
import argparse, json, re, sys, time, requests, collections as C
ap = argparse.ArgumentParser(); ap.add_argument("--model", default=""); ap.add_argument("--n", type=int, default=30); a = ap.parse_args()
key = ""
for l in open("../.env").read().splitlines():
    if l.startswith("GROQ_API_KEY="): key = l.split("=", 1)[1].strip().strip('"')
if not key: sys.exit("GROQ_API_KEY missing in ../.env")
H = {"Authorization": "Bearer " + key}; BASE = "https://api.groq.com/openai/v1"
if not a.model:
    ms = sorted(m["id"] for m in requests.get(BASE + "/models", headers=H, timeout=30).json()["data"])
    print("available models:", ms); sys.exit("pass --model <id> (pick a large multilingual chat model from the list)")
GEN = open("21_make_parallel.py", encoding="utf-8").read().split('GEN = """')[1].split('"""')[0]
CHK = open("21_make_parallel.py", encoding="utf-8").read().split('CHK = """')[1].split('"""')[0]
def ask(p, mt):
    for t in range(6):
        r = requests.post(BASE + "/chat/completions", headers=H, timeout=60, json={"model": a.model, "max_tokens": max(mt, 2500), "temperature": 0.2, **({"reasoning_effort": "low"} if "gpt-oss" in a.model else {}), "messages": [{"role": "user", "content": p}]})
        if r.status_code == 429: time.sleep(5 * (t + 1)); continue
        r.raise_for_status(); return re.sub(r"<think>.*?</think>", "", r.json()["choices"][0]["message"]["content"] or "", flags=re.S).strip()
    raise RuntimeError("rate limited")
bn = lambda s: len(re.findall(r"[ঀ-৿]", s)); la = lambda s: len(re.findall(r"[A-Za-z]", s))
ok_script = {"code_mixed": lambda s: bn(s) > 0 and la(s) > 0, "romanized": lambda s: bn(s) == 0 and la(s) > 0}
rows = [json.loads(l) for l in open("data/parallel_cache.jsonl", encoding="utf-8")][:a.n]
st = C.Counter(); out = []
for r in rows:
    try:
        j = json.loads(re.search(r"\{.*\}", ask(GEN % r["bangla"], 1500), re.S).group(0))
    except Exception as e:
        st["parse_or_api_fail"] += 1; continue
    rec = {"doc_id": r["doc_id"], "bangla": r["bangla"], "haiku": r["gen"], "groq": j, "groq_kept": {}}
    for s in ("code_mixed", "romanized"):
        v = j.get(s, "")
        st[s + "_script_ok"] += bool(v) and ok_script[s](v)
        try: y = ask(CHK % (r["bangla"], v), 20).lower().startswith("yes") if v else False
        except Exception: y = False
        st[s + "_check_yes"] += y
        if y and ok_script[s](v): rec["groq_kept"][s] = v
    st["docs"] += 1; out.append(rec)
with open("data/groq_pilot_" + a.model.replace("/", "_") + ".jsonl", "w", encoding="utf-8") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("model:", a.model, dict(st))
print("Haiku kept both on these docs:", sum(1 for r in rows if len(r["kept"]) == 2), "of", len(rows))
for r in out[:5]: print("\nBN:", r["bangla"], "\n HAIKU MX:", r["haiku"].get("code_mixed"), "\n GROQ  MX:", r["groq"].get("code_mixed"), "\n HAIKU RM:", r["haiku"].get("romanized"), "\n GROQ  RM:", r["groq"].get("romanized"))
