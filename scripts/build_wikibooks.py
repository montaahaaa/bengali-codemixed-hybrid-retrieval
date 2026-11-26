"""Add Bengali Wikibooks as a corpus source (CC BY-SA 4.0). Usage: python scripts/build_wikibooks.py
Uses the official dump (bnwikibooks-latest-pages-articles.xml.bz2, ~19 MB; the API rate-limits page-by-page fetching),
strips wikitext with mwparserfromhell, chunks like Wikipedia and writes data/corpus/corpus_wikibooks.jsonl."""
import bz2, json, sys, urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
import mwparserfromhell
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunking import chunk_article

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/wikibooks"; RAW.mkdir(parents=True, exist_ok=True)
DUMP = RAW / "bnwikibooks-latest-pages-articles.xml.bz2"
OUT = ROOT / "data/corpus/corpus_wikibooks.jsonl"
URL = "https://dumps.wikimedia.org/bnwikibooks/latest/bnwikibooks-latest-pages-articles.xml.bz2"
UA = "bn-code-mixed-retrieval-thesis/0.1 (non-commercial academic research)"

def plain(wt):
    code = mwparserfromhell.parse(wt)
    for t in code.filter_templates(recursive=False): 
        try: code.remove(t)
        except ValueError: pass
    s = code.strip_code(normalize=True, collapse=True)
    return "\n".join(l.strip() for l in s.split("\n") if l.strip())

def main():
    if not DUMP.exists():
        urllib.request.urlretrieve(URL, DUMP) if False else DUMP.write_bytes(urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": UA}), timeout=120).read())
    n_art = n_pass = n_all = 0
    with open(OUT, "w", encoding="utf-8") as fo, bz2.open(DUMP) as f:
        for _, el in ET.iterparse(f):
            if not el.tag.endswith("}page"): continue
            ns = el.find("{*}ns").text
            if ns == "0" and el.find("{*}redirect") is None:
                n_all += 1
                rev = el.find("{*}revision")
                text = plain(rev.find("{*}text").text or "")
                if len(text.split()) >= 40 and "{{" not in text and "<ref" not in text:
                    pid, title = el.find("{*}id").text, el.find("{*}title").text
                    n_art += 1
                    for i, ps in enumerate(chunk_article(text, "bn")):
                        fo.write(json.dumps({"passage_id": f"bn-wb{pid}-{i}", "lang": "bn", "title": title, "text": ps,
                                             "source": "bn.wikibooks.org", "licence": "CC BY-SA 4.0", "source_article_id": f"wb{pid}",
                                             "revision": rev.find("{*}id").text, "snapshot": rev.find("{*}timestamp").text}, ensure_ascii=False) + "\n")
                        n_pass += 1
            el.clear()
    print("pages", n_all, "articles kept", n_art, "passages", n_pass)

if __name__ == "__main__":
    main()
