"""Rerun BM25 / normaliser / dense / hybrid on the 145 human-reviewed needs: Romanized original vs reviewer-corrected text.
Bangla and mixed versions of the same needs are scored as unreviewed references. Same systems as 16_normaliser.py."""
src = open("16_normaliser.py", encoding="utf-8").read().split("qrels = {")[0]
exec(src)
import csv as _csv
qrels = {r["query-id"]: idx[r["corpus-id"]] for r in _csv.DictReader(open("data/qrels.csv"))}
rv = list(_csv.DictReader(open("data/review/romanized_review_results.csv", encoding="utf-8-sig")))
qmeta = {json.loads(l)["_id"]: json.loads(l) for l in open("data/queries.jsonl", encoding="utf-8")}
rows = []                                           # (label, qid, text)
sheet = {r["need_id"]: r for r in _csv.DictReader(open("data/review/review_sheet.csv", encoding="utf-8-sig"))}
for r in rv:
    s = sheet[r["need_id"]]
    rows.append(("romanized_original", r["q_romanized_id"], r["original"], r["verdict"]))
    rows.append(("romanized_reviewed", r["q_romanized_id"], r["reviewed_text"], r["verdict"]))
    rows.append(("bangla_unreviewed", s["q_bangla_id"], s["q_bangla"], r["verdict"]))
    rows.append(("mixed_unreviewed", s["q_mixed_id"], s["q_mixed"], r["verdict"]))
SYS = ["bm25", "bm25_norm_K3", "dense_e5", "convex0.7_raw", "convex0.7_norm"]
res = C.defaultdict(lambda: C.defaultdict(list)); per = {}
def mm(x): return (x - x.min()) / (x.max() - x.min() + 1e-9)
def nd(order, gold):
    pos = np.where(np.asarray(order[:100]) == gold)[0]; r = int(pos[0]) if len(pos) else None
    return (1.0 if r is not None and r < 10 else 0.0), (1 / math.log2(r + 2) if r is not None and r < 10 else 0.0)
def bms(t):
    docs, sc = ret.retrieve([t], k=1000, show_progress=False); b = np.zeros(len(ids), dtype="float32"); b[docs[0]] = sc[0]; return b
Q = model.encode(["query: " + t for _, _, t, _ in rows], batch_size=64, normalize_embeddings=True, convert_to_tensor=True).cpu().float()
D = (Q @ P.T).numpy()
for k, (lab, qid, text, verdict) in enumerate(rows):
    gold = qrels[qid]; d = D[k]; b0 = bms(tok(text)); b3 = bms(expand(text, 3))
    def cv(b): c = np.array(sorted(set(np.argsort(-b)[:1000]) | set(np.argsort(-d)[:1000]))); return c[np.argsort(-(0.7 * mm(d[c]) + 0.3 * mm(b[c])))]
    orders = {"bm25": np.argsort(-b0), "bm25_norm_K3": np.argsort(-b3), "dense_e5": np.argsort(-d), "convex0.7_raw": cv(b0), "convex0.7_norm": cv(b3)}
    for s, o in orders.items():
        r10, n10 = nd(list(o), gold); per[(lab, qid, s)] = n10
        for key in (lab, lab + "|" + verdict): res[(key, s)]["R@10"].append(r10); res[(key, s)]["nDCG@10"].append(n10)
print(f"{'group':34s}{'n':>5s}" + "".join(f"{x:>16s}" for x in SYS) + "   (nDCG@10 / R@10)")
out = {}
for key in ["bangla_unreviewed", "mixed_unreviewed", "romanized_original", "romanized_reviewed", "romanized_original|change", "romanized_reviewed|change", "romanized_original|no change"]:
    n = len(res[(key, "bm25")]["R@10"]); row = f"{key:34s}{n:5d}"
    for s in SYS:
        r = res[(key, s)]; out[f"{key}|{s}"] = {k: float(np.mean(v)) for k, v in r.items()}; row += f"{np.mean(r['nDCG@10']):.3f}/{np.mean(r['R@10']):.3f}".rjust(16)
    print(row)
rng = np.random.default_rng(42); qids = [q for l, q, _, _ in rows if l == "romanized_original"]
for a, b in (("bm25_norm_K3", "bm25"), ("convex0.7_norm", "convex0.7_raw"), ("convex0.7_norm", "dense_e5")):
    for lab in ("romanized_original", "romanized_reviewed"):
        d = np.array([per[(lab, q, a)] - per[(lab, q, b)] for q in qids]); bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
        print(f"paired nDCG@10 {lab:20s} {a} - {b}: {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]  n={len(d)}")
json.dump(out, open("data/reviewed_rerun_results.json", "w"), indent=1)
