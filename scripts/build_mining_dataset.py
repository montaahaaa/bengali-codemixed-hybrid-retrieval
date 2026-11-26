"""Merge the Facebook queries (project owner's file) and the YouTube pilot comments into one unlabelled
mining dataset for phase 2.3. Usage: python scripts/build_mining_dataset.py
Output: data/processed/mining_pilot.jsonl (git-ignored, PRIVATE: contains platform text, not for release;
YouTube rows must be deleted by their delete_after date) and data/annotations/pilot/mining_pilot_stats.json
(counts only). Cleaning is minimal on purpose: whitespace, exact duplicates, empty/1-token rows.
Real-information-need and answerability are decided by hand in 2.3; `question_like` is only a hint."""
import csv, json, re, unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FB = ROOT / "data collection/facebook_queries_only.csv"
YT = ROOT / "data/raw/youtube/comments.jsonl"
OUT = ROOT / "data/processed/mining_pilot.jsonl"
STATS = ROOT / "data/annotations/pilot/mining_pilot_stats.json"
Q_WORDS = re.compile(r"\b(ki|kि|kobe|kothay|kothai|kemon|kivabe|kibhabe|kemne|koto|keno|kon|kar|ke)\b|কি|কী|কিভাবে|কীভাবে|কেন|কত|কোথায়|কখন|কবে|কোন|কেমন|কারা|কাকে|\?|？", re.I)


PRAISE = re.compile(r"ধন্যবাদ|থ্যাংক|ধন্যবাদ|জাযাকাল|জাজাকাল|আল্লাহ|আলহামদুলিল|ভালো লাগ|সুন্দর ভিডিও|thank|great video|nice video|love from|subscribe|helpful|waalaikum|whatsapp|what's up|https?://|www\\.|\\b01\\d{9}\\b", re.I)


def likely_need(t, qlike):
    """Heuristic pre-filter for a real information need: question-like, at least 4 words, not praise/ad/link."""
    return qlike and len(t.split()) >= 4 and not PRAISE.search(t)


def script_of(t):
    bn = sum("ঀ" <= c <= "৿" for c in t)
    la = sum(c.isascii() and c.isalpha() for c in t)
    if bn + la == 0:
        return "other"
    r = bn / (bn + la)
    return "bn" if r > 0.9 else "latin" if r < 0.1 else "mixed"


def clean(t):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", t)).strip()


rows, seen = [], set()
def add(source, src_id, text, domain, extra):
    text = clean(text)
    key = text.casefold()
    if len(text.split()) < 2 or key in seen:
        return False
    seen.add(key)
    rows.append({"id": f"{source[:2]}-{src_id}", "source": source, "domain": domain, "text": text,
                 "script": script_of(text), "question_like": bool(Q_WORDS.search(text)), **extra})
    rows[-1]["likely_need"] = likely_need(text, rows[-1]["question_like"])
    return True

dropped = Counter()
with FB.open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if not add("facebook", r["query_id"], r["query_text"], None, {}):
            dropped["facebook"] += 1
with YT.open(encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if not add("youtube", r["comment_id"], r["text"], r["domain"],
                   {"video_id": r["video_id"], "delete_after": r["delete_after"]}):
            dropped["youtube"] += 1

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
stats = {"total": len(rows), "dropped_empty_short_or_duplicate": dict(dropped),
         "by_source": Counter(r["source"] for r in rows),
         "by_source_script": Counter(f'{r["source"]}/{r["script"]}' for r in rows),
         "by_source_likely_need": Counter(f'{r["source"]}/{r["likely_need"]}' for r in rows),
         "by_source_question_like": Counter(f'{r["source"]}/{r["question_like"]}' for r in rows),
         "youtube_by_domain": Counter(r["domain"] for r in rows if r["source"] == "youtube")}
STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(stats, ensure_ascii=False, indent=2))
