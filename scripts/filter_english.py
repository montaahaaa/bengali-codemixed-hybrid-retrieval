"""Keep English Wikipedia articles that match data/corpus/en_filter.json.

Downloads one parquet shard at a time, filters it and appends matches to
data/raw/wikipedia/en_filtered.jsonl. Shards are kept on disk (about 11 GB), so a
changed filter can be rerun without downloading again. Every article's id, title
and keep/drop decision goes to data/raw/wikipedia/en_decisions.parquet, so any
title can be looked up offline. Counts go to data/corpus/en_filter_counts.json.
"""
import argparse, json, re
from pathlib import Path
import pandas as pd
from download_wikipedia import shards, fetch, ROOT

CFG = json.loads((ROOT / "data/corpus/en_filter.json").read_text())


def compile_terms(terms):
    return re.compile(r"\b(" + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)) + r")\b", re.I)


DOMAIN_RE = {d: compile_terms(t) for d, t in CFG["domains"].items()}
ALL_RE = compile_terms([t for ts in CFG["domains"].values() for t in ts])
REGION_RE = compile_terms(CFG["region_terms"])


def classify(title, text):
    """Return (kept, domains_matched)."""
    lead = text[: CFG["lead_chars"]]
    in_title = {d for d, r in DOMAIN_RE.items() if r.search(title)}
    lead_terms = {m.group(0).lower() for m in ALL_RE.finditer(lead)}
    in_lead = {d for d, r in DOMAIN_RE.items() if r.search(lead)}
    if in_title:
        return True, sorted(in_title)
    if len(lead_terms) >= CFG["min_lead_terms"]:
        return True, sorted(in_lead)
    if len(lead_terms) >= CFG["min_lead_terms_with_region"] and REGION_RE.search(lead):
        return True, sorted(in_lead)
    return False, []


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--max-shards", type=int, default=None, help="for testing")
    a = ap.parse_args()
    out = ROOT / "data/raw/wikipedia/en_filtered.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = {"seen": 0, "kept": 0, "per_domain": {d: 0 for d in CFG["domains"]}}
    decisions = []
    with out.open("w", encoding="utf-8") as fo:
        for s in shards("en", a.snapshot)[: a.max_shards]:
            path = Path(fetch(s))
            df = pd.read_parquet(path)
            for r in df.itertuples():
                counts["seen"] += 1
                ok, doms = classify(r.title, r.text)
                decisions.append((r.id, r.title, ok))
                if ok:
                    counts["kept"] += 1
                    for d in doms:
                        counts["per_domain"][d] += 1
                    fo.write(json.dumps({"id": r.id, "title": r.title, "text": r.text, "domains": doms}, ensure_ascii=False) + "\n")
            print(s, counts["seen"], counts["kept"], flush=True)
    pd.DataFrame(decisions, columns=["id", "title", "kept"]).to_parquet(ROOT / "data/raw/wikipedia/en_decisions.parquet")
    (ROOT / "data/corpus/en_filter_counts.json").write_text(json.dumps(counts, indent=2))
