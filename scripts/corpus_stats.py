"""Corpus stats -> data/corpus/stats.json, and a seeded sample of 20 passages to read."""
import json, random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
rows = [json.loads(l) for l in open(ROOT / "data/corpus/corpus.jsonl", encoding="utf-8")]
stats = {"total_passages": len(rows), "seed": SEED, "snapshot": rows[0]["snapshot"], "by_lang": {}}
for lang in ("bn", "en"):
    r = [x for x in rows if x["lang"] == lang]
    w = np.array([len(x["text"].split()) for x in r])
    stats["by_lang"][lang] = {
        "passages": len(r),
        "articles": len({x["source_article_id"] for x in r}),
        "words_min": int(w.min()), "words_median": float(np.median(w)), "words_max": int(w.max()),
        "words_p5": float(np.percentile(w, 5)), "words_p95": float(np.percentile(w, 95)),
        "share_100_150": round(float(((w >= 100) & (w <= 150)).mean()), 4),
        "share_under_100": round(float((w < 100).mean()), 4),
        "share_over_150": round(float((w > 150).mean()), 4),
    }
assert len({x["passage_id"] for x in rows}) == len(rows), "duplicate passage ids"
(ROOT / "data/corpus/stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
print(json.dumps(stats, indent=2))
random.seed(SEED)
sample = random.sample(rows, 20)
(ROOT / "data/corpus/reading_sample.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in sample))
