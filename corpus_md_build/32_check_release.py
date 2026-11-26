"""Recompute every dataset statistic that the thesis states. Read-only. Usage (from corpus_md_build): python 32_check_release.py .."""
import csv, hashlib, json, re, sys, statistics as st, collections as C
from pathlib import Path

ROOT = Path(sys.argv[1]); SUB = ROOT / "submission"; B = ROOT / "corpus_md_build" / "data"; R = ROOT / "review"
csv.field_size_limit(10**9)
rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8-sig")))
Q = rd(SUB / "queries.csv"); QR = rd(SUB / "qrels.csv"); CO = rd(SUB / "corpus.csv")
bn = lambda s: len(re.findall(r"[ঀ-৿]", s)); la = lambda s: len(re.findall(r"[A-Za-z]", s))
tok = lambda t: [w for w in re.split(r"[\s\.,;:!?()\[\]\"'“”‘’।|/\\\-–—]+", t.lower()) if w]
gold = {r["query_id"]: r["passage_id"] for r in QR}
print("queries", len(Q), "unique ids", len({q["query_id"] for q in Q}), "qrels", len(QR), "corpus", len(CO))
print("all gold present", set(gold.values()) <= {c["passage_id"] for c in CO}, "distinct gold", len(set(gold.values())))
for pt in "PG":
    S = [q for q in Q if q["query_id"][0] == pt]
    print(pt, len(S), "passages", len({gold[q["query_id"]] for q in S}))
    print("  script x split", sorted(C.Counter((q["script"], q["split"]) for q in S).items()))
print("script", C.Counter(q["script"] for q in Q), "split", C.Counter(q["split"] for q in Q))
print("domain", C.Counter(q["domain"] for q in Q).most_common())
for s in ("bangla", "mixed", "romanized"):
    L = [len(q["query"].split()) for q in Q if q["query_id"] and q["script"] == s]
    print("words", s, round(st.mean(L), 2), min(L), max(L))
print("words all", round(st.mean(len(q["query"].split()) for q in Q), 2))
pl = [len(c["text"].split()) for c in CO]
print("passage words mean", round(st.mean(pl), 1), "median", st.median(pl), "min", min(pl), "max", max(pl), "in 80-300", sum(80 <= x <= 300 for x in pl))
sp = C.defaultdict(set)
for q in Q: sp[gold[q["query_id"]]].add(q["split"])
print("passages in >1 split", sum(len(v) > 1 for v in sp.values()))
print("empty", sum(not q["query"].strip() for q in Q), "under 3 words", sum(len(q["query"].split()) < 3 for q in Q))
print("exact dup", len(Q) - len({q["query"] for q in Q}))
norm = lambda s: " ".join(tok(s))
print("normalised dup", len(Q) - len({norm(q["query"]) for q in Q}))
# near duplicates within script and split (Jaccard >= 0.7), inverted index
for thr in (0.7,):
    groups = C.defaultdict(list)
    for q in Q: groups[(q["script"], q["split"])].append((q["query_id"], frozenset(tok(q["query"]))))
    pairs = []
    for k, L in groups.items():
        inv = C.defaultdict(list)
        for i, (qid, t) in enumerate(L):
            cnt = C.Counter(j for w in t for j in inv[w])
            for j, n in cnt.items():
                if n / len(t | L[j][1]) >= thr: pairs.append((qid, L[j][0]))
            for w in t: inv[w].append(i)
    print("near-dup pairs J>=0.7", len(pairs), pairs[:8])
def script_ok(s, q):
    b, l = bn(q), la(q)
    return (b > 0 and l / (b + l) < 0.05) if s == "bangla" else (b == 0 and l > 0) if s == "romanized" else (b > 0 and l > 0)
bad = [q["query_id"] for q in Q if not script_ok(q["script"], q["query"])]
print("script label mismatches", len(bad), bad[:10])
P = C.defaultdict(dict)
for q in Q:
    if q["query_id"][0] == "P": P[q["query_id"][:-2]][q["query_id"][-1]] = q
print("P needs", len(P), "complete triples", sum(len(v) == 3 for v in P.values()), "identical mixed==bangla", sum(v["b"]["query"] == v["m"]["query"] for v in P.values() if "m" in v))
print("P romanized non-ascii", sum(any(ord(ch) > 127 for ch in v["r"]["query"]) for v in P.values()))
# review sheets
OV = json.load(open(B / "review_overrides.json", encoding="utf-8"))
rel = {q["query_id"]: q["query"] for q in Q}
for name in ("review_sheet_A.csv", "review_sheet_B.csv"):
    S = rd(R / name)
    print(name, len(S), "needs", len({r["need_id"] for r in S}))
    for sc in ("bangla", "mixed", "romanized"):
        T = [r for r in S if r["script"] == sc]
        for col in ("natural_yes_no", "answerable_from_passage_yes_no", "same_meaning_as_bangla_yes_no"):
            c = C.Counter(r[col].strip().lower() for r in T); print("  ", sc, col[:12], dict(c))
    print("   corrections", sum(bool(r["corrected_query_if_needed"].strip()) for r in S), "notes", sum(bool(r["note"].strip()) for r in S))
    match = sum(rel.get(r["query_id"]) in (r["query"], r["corrected_query_if_needed"].strip()) for r in S)
    print("   sheet text matches released text (before or after correction):", match, "/", len(S))
    miss = [r["query_id"] for r in S if rel.get(r["query_id"]) not in (r["query"], r["corrected_query_if_needed"].strip())]
    print("   not matching:", miss[:10])
    dm = sum(1 for r in S if r["query_id"] in gold and r["passage"][:200] == next((c["text"][:200] for c in CO if c["passage_id"] == gold[r["query_id"]]), None)) if name.endswith("A.csv") else None
    if dm is not None: print("   sheet passage equals released gold passage:", dm, "/", len(S))
    print("   domains", C.Counter({r["need_id"] for r in S} and [next((q["domain"] for q in Q if q["query_id"] == r["query_id"]), "?") for r in S if r["script"] == "bangla"]))
A, Bb = rd(R / "review_sheet_A.csv"), {r["query_id"]: r for r in rd(R / "review_sheet_B.csv")}
cols = ["natural_yes_no", "answerable_from_passage_yes_no", "same_meaning_as_bangla_yes_no", "corrected_query_if_needed", "note"]
print("B rows identical to A on all judgement columns:", sum(all(r[c] == Bb[r["query_id"]][c] for c in cols) for r in A if r["query_id"] in Bb), "/", len(Bb))
print("overrides", len(OV), "applied in release", sum(rel.get(k) == v for k, v in OV.items()))
print("notes (A):")
for r in A:
    if r["note"].strip(): print("  ", r["query_id"], "|", r["natural_yes_no"], r["answerable_from_passage_yes_no"], "|", r["note"])
F = rd(R / "flagged_queries.csv"); print("flagged", len(F), C.Counter(f["reason"] for f in F), C.Counter(f["query_id"][0] for f in F))
# G reviewed
V1 = [json.loads(l) for l in open(B / "queries.jsonl", encoding="utf-8")]
g = [q for q in V1 if q["_id"] in rel and "human_reviewed" in q["metadata"]["source"]]
print("G reviewed in release", len(g), C.Counter(q["metadata"]["script"] for q in g), "corrected", C.Counter(q["metadata"]["script"] for q in g if q["metadata"].get("text_original") not in (None, q["text"])))
print("G released text equals v1 text", sum(rel[q["_id"]] == q["text"] for q in V1 if q["_id"] in rel and q["_id"][0] == "G"), sum(q["_id"] in rel and q["_id"][0] == "G" for q in V1))
print("v1 total", len(V1), C.Counter(q["metadata"]["script"] for q in V1))
g1 = [json.loads(l) for l in open(B / "gen_cache.jsonl", encoding="utf-8")]
print("gen_cache passages", len(g1), "usable", sum(x["usable"] for x in g1), "kept queries", sum(len(x["kept"]) for x in g1))
for f in ("parallel_cache.jsonl", "parallel_cache_groq.jsonl"):
    L = [json.loads(l) for l in open(B / f, encoding="utf-8")]
    print(f, len(L), "both kept", sum(len(x["kept"]) == 2 for x in L), "mixed kept", sum("code_mixed" in x["kept"] for x in L), "rom kept", sum("romanized" in x["kept"] for x in L))
v2 = [json.loads(l) for l in open(B / "v2" / "queries.jsonl", encoding="utf-8")]
print("v2 queries", len(v2), "not released", [q["_id"] for q in v2 if q["_id"] not in rel][:20])
w = C.Counter(c["passage_id"] for c in CO); print("dup passage ids", sum(v > 1 for v in w.values()), "dup passage text", len(CO) - len({c["text"] for c in CO}))
