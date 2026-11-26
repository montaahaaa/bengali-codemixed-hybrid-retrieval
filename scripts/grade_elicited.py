"""Phase 2.4 helper: breakdown + answerability check of the elicited queries (writer_A/B/C.csv).
Usage: python scripts/grade_elicited.py
Reads data/annotations/pilot/elicitation/writer_*.csv. Writes data/processed/elicited_graded.csv (private text, git-ignored)
and data/annotations/pilot/elicited_stats.json (counts only). Grading is the same proxy as D-035: BM25 top-10 over the
1.1M-passage pilot corpus, reranked with BAAI/bge-reranker-v2-m3; score >=0.60 -> 2, >=0.20 -> 1, else 0. Uncalibrated.
A duplicate query (same text across writers) is graded once."""
import csv, json, re, sys, collections as C
from pathlib import Path
import numpy as np, torch
from sentence_transformers import CrossEncoder
sys.path.insert(0, str(Path(__file__).resolve().parent))
from find_candidates import load, tok

ROOT = Path(__file__).resolve().parent.parent
E = ROOT / "data/annotations/pilot/elicitation"
BN, LAT = re.compile(r"[ঀ-৿]"), re.compile(r"[A-Za-z]")
rows = []
for w in "ABC":
    for r in csv.DictReader(open(E / f"writer_{w}.csv", encoding="utf-8-sig")):
        q = (r["your_query"] or "").strip()
        if q: rows.append(dict(writer=w, row=r["row"], domain=r["domain"], topic=r["topic"], how=r["how_written"].strip(), query=q))
words = lambda q: re.findall(r"[ঀ-৿]+|[A-Za-z]+(?:'[a-z]+)?|\d+", q)
for r in rows:
    ws = words(r["query"]); nb = sum(bool(BN.search(x)) for x in ws); nl = sum(bool(LAT.search(x)) for x in ws)
    r["n_words"], r["n_bn"], r["n_latin"] = len(ws), nb, nl
    r["script"] = "bn" if nl == 0 else "latin" if nb == 0 else "mixed"
    r["latin_share"] = round(nl / max(1, nb + nl), 2)
uniq, dup = {}, {}
for r in rows:
    if r["query"] in uniq: dup.setdefault(r["query"], [uniq[r["query"]]["writer"] + uniq[r["query"]]["row"]]).append(r["writer"] + r["row"])
    else: uniq[r["query"]] = r
U = list(uniq.values())

bm, ids = load()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
ce = CrossEncoder("BAAI/bge-reranker-v2-m3", device=dev, max_length=320, activation_fn=torch.nn.Sigmoid(), model_kwargs={"torch_dtype": torch.float16})
for k, r in enumerate(U, 1):
    s = bm.get_scores(tok(r["query"])); top = np.argsort(-s)[:10]
    c = [(ids[i][0], ids[i][1], ids[i][2]) for i in top if s[i] > 0]
    if not c: r.update(grade=0, score=0.0, best_id="", best_title="", best_text=""); continue
    sc = ce.predict([(r["query"], f"{t}. {x}") for _, t, x in c], batch_size=10, show_progress_bar=False)
    b = int(np.argmax(sc)); r.update(grade=2 if sc[b] >= .6 else 1 if sc[b] >= .2 else 0, score=round(float(sc[b]), 3),
                                     best_id=c[b][0], best_title=c[b][1], best_text=c[b][2][:400])
    print(k, len(U), r["grade"], r["score"], flush=True)

cols = ["writer", "row", "domain", "topic", "how", "script", "n_words", "n_bn", "n_latin", "latin_share", "grade", "score", "query", "best_id", "best_title", "best_text"]
with open(ROOT / "data/processed/elicited_graded.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, cols, extrasaction="ignore"); w.writeheader(); w.writerows(U)
by = lambda k, rs=U: dict(C.Counter(r[k] for r in rs))
st = dict(entries=len(rows), unique=len(U), duplicates={k: v for k, v in dup.items()} and len(dup),
          by_writer=by("writer", rows), how_written=by("how", rows), script_by_chars=by("script"), by_domain=by("domain"),
          grade=by("grade"), grade_by_domain={d: dict(C.Counter(r["grade"] for r in U if r["domain"] == d)) for d in sorted({r["domain"] for r in U})},
          grade_by_writer={w: dict(C.Counter(r["grade"] for r in U if r["writer"] == w)) for w in "ABC"},
          words_mean=round(float(np.mean([r["n_words"] for r in U])), 1), latin_share_mean=round(float(np.mean([r["latin_share"] for r in U])), 2),
          fully_bn_by_chars=sum(r["script"] == "bn" for r in U))
json.dump(st, open(ROOT / "data/annotations/pilot/elicited_stats.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(st, indent=1, ensure_ascii=False))
