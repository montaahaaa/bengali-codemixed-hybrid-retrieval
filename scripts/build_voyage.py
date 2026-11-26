"""Add Bangladesh-related Wikivoyage pages as a separate corpus source (D-018, D-030).
Usage: python scripts/build_voyage.py
Fetches plain text through the Wikivoyage API (one page per request), saves raw pages to
data/raw/wikivoyage/pages_<lang>.jsonl, chunks them with the same chunker as Wikipedia and
writes data/corpus/corpus_voyage.jsonl. Text is CC BY-SA 4.0; source and licence are kept on
every passage."""
import json, sys, time
import urllib.error, urllib.parse, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunking import chunk_article

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/wikivoyage"
OUT = ROOT / "data/corpus/corpus_voyage.jsonl"
UA = "bn-code-mixed-retrieval-thesis/0.1 (non-commercial academic research)"
LICENCE = "CC BY-SA 4.0"

# English: Bangladesh pages and a few generic pages useful to Bangladesh travellers.
EN_TITLES = [
    "Bangladesh", "Dhaka", "Dhaka Division", "Chittagong", "Chittagong Division", "Cox's Bazar",
    "Rajshahi", "Rajshahi Division", "Rangpur (Bangladesh)", "Rangpur Division", "Khulna", "Sylhet",
    "Barisal", "Mymensingh", "Jessore", "Bagerhat", "Paharpur", "Sundarbans (Bangladesh)", "Saint Martins Island",
    "Banglaband", "Chilahati", "Hili", "Hularhat", "Tamabil", "Teknaf", "Mowdok Mual", "Ashuganj", "Morrelganj",
    "Northeastern Bangladesh", "Southern Bangladesh", "Padna Meghna River Estuarine", "Bengali phrasebook",
    "Ride hailing services", "Mosquitoes", "Common scams",
]


def api(host, **params):
    params.update(format="json", formatversion=2)
    url = f"https://{host}/w/api.php?" + urllib.parse.urlencode(params)
    for i in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            return json.load(urllib.request.urlopen(req, timeout=40))
        except urllib.error.HTTPError as e:
            wait = int(e.headers.get("Retry-After", 0) or 0) or 20 * (i + 1)
            print("retry", i + 1, "HTTP", e.code, "wait", wait, flush=True)
            time.sleep(wait)
        except Exception as e:
            print("retry", i + 1, str(e)[:80], flush=True)
            time.sleep(5 * (i + 1))
    raise RuntimeError(url)


def fetch_page(host, title):
    d = api(host, action="query", prop="extracts|revisions", explaintext=1, redirects=1, titles=title,
            rvprop="ids|timestamp")
    p = d["query"]["pages"][0]
    if p.get("missing") or not p.get("extract"):
        return None
    rev = p["revisions"][0]
    return {"pageid": p["pageid"], "title": p["title"], "text": p["extract"],
            "revid": rev["revid"], "timestamp": rev["timestamp"]}


def bn_titles(limit=150):
    titles, cont = [], {}
    while len(titles) < limit:
        d = api("bn.wikivoyage.org", action="query", list="search", srsearch="বাংলাদেশ", srnamespace=0,
                srlimit=50, **cont)
        titles += [r["title"] for r in d["query"]["search"]]
        titles = titles[:limit]
        if "continue" not in d:
            break
        cont = {"sroffset": d["continue"]["sroffset"]}
        time.sleep(1.5)
    return titles


def fetch_all(lang, host, titles, keyword=None, min_hits=0):
    """Resumable: pages already saved are kept; only missing ones are fetched."""
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"pages_{lang}.jsonl"
    seen = {}
    if path.exists():
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            seen[d["query_title"]] = d
    with open(path, "a", encoding="utf-8") as fo:
        for t in titles:
            if t in seen:
                continue
            p = fetch_page(host, t)
            time.sleep(1.5)
            if not p:
                print("skip (missing/empty):", t)
                p = {"query_title": t, "empty": True}
            else:
                p["query_title"] = t
            seen[t] = p
            fo.write(json.dumps(p, ensure_ascii=False) + "\n")
            fo.flush()
    kept = [d for d in seen.values() if not d.get("empty")
            and "{{" not in d["text"]  # leftover wikitext templates: drop the whole page
            and (not keyword or d["text"].count(keyword) >= min_hits)]
    with open(RAW / f"pages_{lang}_kept.jsonl", "w", encoding="utf-8") as fo:
        for d in kept:
            fo.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(lang, "pages kept", len(kept), "of", len(titles))


def main():
    fetch_all("en", "en.wikivoyage.org", EN_TITLES)
    cached = RAW / "pages_bn.jsonl"  # reuse saved titles so a rebuild needs no network
    t = [json.loads(l)["query_title"] for l in open(cached, encoding="utf-8")] if cached.exists() else bn_titles()
    print("bn search titles", len(t))
    fetch_all("bn", "bn.wikivoyage.org", t, keyword="বাংলাদেশ", min_hits=3)
    n_art = n_pass = 0
    with open(OUT, "w", encoding="utf-8") as fo:
        for lang in ("en", "bn"):
            for line in open(RAW / f"pages_{lang}_kept.jsonl", encoding="utf-8"):
                p = json.loads(line)
                n_art += 1
                text = "\n".join(l.strip("= ").strip() if l.startswith("=") else l for l in p["text"].split("\n"))
                for i, ps in enumerate(chunk_article(text, lang)):
                    fo.write(json.dumps({
                        "passage_id": f"{lang}-wv{p['pageid']}-{i}", "lang": lang, "title": p["title"], "text": ps,
                        "source": f"{lang}.wikivoyage.org", "licence": LICENCE, "source_article_id": f"wv{p['pageid']}",
                        "revision": p["revid"], "snapshot": p["timestamp"]}, ensure_ascii=False) + "\n")
                    n_pass += 1
    print("articles", n_art, "passages", n_pass)


if __name__ == "__main__":
    main()
