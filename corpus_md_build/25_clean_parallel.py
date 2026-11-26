"""Step 2: clean parallel v2 rows. Merges both caches, fixes Romanized characters, drops bad/duplicate queries.
Output: data/parallel_clean.jsonl (one row per passage) and data/parallel_clean_report.json. Re-runnable."""
import json, re, unicodedata, collections
SRC = ["data/parallel_cache.jsonl", "data/parallel_cache_groq.jsonl"]
LOOK = str.maketrans({"е": "e", "р": "p", "а": "a", "о": "o", "с": "c", "х": "x", "‑": "-", "–": "-", "’": "'", "“": '"', "”": '"'})
bn = lambda s: len(re.findall(r"[ঀ-৿]", s)); la = lambda s: len(re.findall(r"[A-Za-z]", s))
def fix_rom(s):
    s = s.translate(LOOK)
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")  # strip diacritics
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"\s+", " ", s).strip()
rep = collections.Counter(); out = {}; seen = set()
for p in SRC:
    for l in open(p, encoding="utf-8"):
        r = json.loads(l); rep["rows_in"] += 1
        if r["doc_id"] in out: rep["dup_doc"] += 1; continue
        k = {}
        m = (r["kept"].get("code_mixed") or "").strip()
        if m:
            m = re.sub(r"\s+", " ", m)
            if bn(m) and la(m) and m != r["bangla"]: k["code_mixed"] = m
            else: rep["mixed_bad"] += 1
        ro = r["kept"].get("romanized")
        if ro:
            f = fix_rom(ro)
            if f != ro: rep["rom_fixed"] += 1
            if bn(f) == 0 and la(f) >= 5 and all(ord(c) < 128 for c in f): k["romanized"] = f
            else: rep["rom_dropped_nonascii"] += 1
        # drop query texts already used by another passage
        for st in list(k):
            key = (st, k[st].lower())
            if key in seen: rep["dup_query"] += 1; del k[st]
            else: seen.add(key)
        rep["has_" + "+".join(sorted(k)) if k else "has_none"] += 1
        out[r["doc_id"]] = {**r, "kept": k}
rows = [r for r in out.values() if r["kept"]]
with open("data/parallel_clean.jsonl", "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
rep["passages_out"] = len(rows); rep["full_triples"] = sum(len(r["kept"]) == 2 for r in rows)
rep["queries_out"] = len(rows) + sum(len(r["kept"]) for r in rows)
json.dump(dict(rep), open("data/parallel_clean_report.json", "w"), indent=1)
print(dict(rep))
