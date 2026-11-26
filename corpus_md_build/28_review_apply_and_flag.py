"""Step 4b: (1) turn reviewer-1 corrections into data/review_overrides.json (applied by 26_build_submission.py),
(2) flag queries that point to something unnamed ("this school", "তার") with a simple rule, and test the rule on the reviewed sample."""
import csv, json, re, collections as C
bn = lambda s: len(re.findall(r"[ঀ-৿]", s)); la = lambda s: len(re.findall(r"[A-Za-z]", s))
okscript = {"bangla": lambda s: bn(s) > 0 and la(s) / max(bn(s) + la(s), 1) < 0.05, "mixed": lambda s: bn(s) > 0 and la(s) > 0, "romanized": lambda s: bn(s) == 0 and la(s) > 0}
rows = list(csv.DictReader(open("../review/review_sheet_A.csv", encoding="utf-8-sig")))
ov = {}; bad = []
for r in rows:
    c = r["corrected_query_if_needed"].strip()
    if not c: continue
    if okscript[r["script"]](c): ov[r["query_id"]] = c
    else: bad.append(r["query_id"])
json.dump(ov, open("data/review_overrides.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("corrections ready:", len(ov), "| script-mismatch (not applied):", bad)
# --- flagging rule: pointer words that need an unnamed referent
BN = re.compile(r"(^|[\s\"'(])(এই|ঐ|ওই|উক্ত|তার|তাঁর|তিনি|তাকে|তাঁকে|তাদের|এটি|এটা|এর)(?=[\s?।,]|[ঀ-৿]|$)")
EN = re.compile(r"\b(this|that|these|those|his|her|their|he|she|they)\b", re.I)
RO = re.compile(r"\b(ei|oi|ey|tar|tader|tini|take|eti|eta|er)\b", re.I)
def flag(q, sc):
    mt = BN.search(q) if sc != "romanized" else None
    if mt: return "demonstrative" if mt.group(2) in ("এই", "ঐ", "ওই", "উক্ত") else "pronoun"
    if sc == "mixed" and EN.search(q): return "en_pointer"
    return ""
if __name__ == "__main__":
    # test on reviewed sample: ground truth = passages where reviewer wrote the entity was missing (by hand from notes)
    truth = {"P00189", "P00173", "P01014", "P00824", "P00081", "P00671", "P00286", "P01105", "P00658"}
    bang = {r["need_id"]: r["query"] for r in rows if r["script"] == "bangla"}
    pred = {n for n, q in bang.items() if flag(q, "bangla")}
    tp = len(pred & truth); print(f"rule on 34 reviewed Bangla queries: flagged {len(pred)}, true {len(truth)}, caught {tp}, precision {tp/max(len(pred),1):.2f}, recall {tp/len(truth):.2f}")
    print("missed:", sorted(truth - pred)); print("false alarms:", sorted(pred - truth))
    for n in sorted(pred - truth): print("  ", n, bang[n])
    # flag list for the unreviewed queries (does not change the dataset)
    Q = list(csv.DictReader(open("../submission/queries.csv", encoding="utf-8"))); rev = {r["query_id"] for r in rows}
    out = [(q["query_id"], q["script"], flag(q["query"], q["script"]), q["query"]) for q in Q if q["query_id"] not in rev and flag(q["query"], q["script"])]
    w = csv.writer(open("../review/flagged_queries.csv", "w", newline="", encoding="utf-8-sig")); w.writerow(["query_id", "script", "reason", "query"]); w.writerows(out)
    print("flagged", len(out), "of", len(Q) - len(rev), C.Counter(o[2] for o in out), C.Counter(o[1] for o in out), "| P:", sum(o[0][0] == "P" for o in out), "G:", sum(o[0][0] == "G" for o in out))
