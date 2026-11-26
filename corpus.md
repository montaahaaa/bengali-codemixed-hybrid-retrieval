Copy and run these commands **one block at a time**. They will create the folders, install dependencies, download Bengali Wikipedia, crawl the approved websites, and build the final `corpus.jsonl`. Hugging Face `load_dataset()` can load the Wikipedia Parquet files, and Trafilatura extracts main page text from crawled HTML. [huggingface](https://huggingface.co/docs/datasets/v4.8.2/en/loading)

## 1. Create project folders

```bash
mkdir -p bengali_code_mixed_rag_benchmark
cd bengali_code_mixed_rag_benchmark

mkdir -p data/raw data/scripts data/corpus
```

## 2. Create and activate environment

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

# Windows Command Prompt
# .venv\Scripts\activate.bat
```

## 3. Install required packages

```bash
pip install --upgrade pip

pip install datasets requests beautifulsoup4 lxml trafilatura tqdm pandas pyyaml
```

## 4. Save the Wikipedia downloader

Create a file named `01_download_wikipedia.py` and paste this code:

```python
from datasets import load_dataset
import json
import os

OUT_PATH = "data/raw/wikipedia.jsonl"
DATASET_NAME = "sagorsarker/bangla-wikipedia"
CONFIG_NAME = "20250620"

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

print("Downloading Bengali Wikipedia dataset...")
dataset = load_dataset(DATASET_NAME, CONFIG_NAME, split="train")

count = 0

with open(OUT_PATH, "w", encoding="utf-8") as out_file:
    for row in dataset:
        title = str(row.get("title", "")).strip()
        text = str(row.get("text", "")).strip()

        if not title or len(text.split()) < 100:
            continue

        record = {
            "source": "bengali_wikipedia",
            "source_id": str(row.get("id", "")),
            "title": title,
            "url": str(row.get("url", "")),
            "text": text,
            "language": "bn"
        }

        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1

print(f"Saved {count} Bengali Wikipedia articles to {OUT_PATH}")
```

Run it:

```bash
python 01_download_wikipedia.py
```

## 5. Save the web crawler

Create a file named `02_crawl_web.py` and paste this code:

```python
import json
import os
import time
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
import trafilatura
from tqdm import tqdm

USER_AGENT = "BengaliCodeMixedRAGBenchmark/1.0 (+research; contact: your_email@example.com)"
DELAY_SECONDS = 1.5
MAX_PAGES_PER_DOMAIN = 1000
MIN_WORDS = 80

SOURCES = [
    "https://www.dghs.gov.bd/",
    "https://moedu.gov.bd/",
    "https://www.dae.gov.bd/",
    "https://bangladesh.gov.bd/",
    "https://mopa.gov.bd/",
    "https://mofa.gov.bd/",
    "https://mof.gov.bd/",
    "https://moj.gov.bd/",
    "https://mowr.gov.bd/",
    "https://moef.gov.bd/",
    "https://www.du.ac.bd/",
    "https://www.buet.ac.bd/",
    "https://www.bau.edu.bd/",
    "https://www.juniv.edu/",
    "https://www.sust.edu/",
    "https://www.nu.edu.bd/",
    "https://www.bou.ac.bd/",
    "https://www.cu.ac.bd/",
    "https://www.ru.ac.bd/",
    "https://www.ku.ac.bd/",
    "https://www.prothomalo.com/",
    "https://www.kalerkantho.com/",
    "https://www.ittefaq.com.bd/",
    "https://www.jugantor.com/",
    "https://www.samakal.com/",
    "https://www.tbsnews.net/",
    "https://www.thedailystar.net/",
    "https://www.dhakatribune.com/",
    "https://www.daraz.com.bd/",
    "https://www.chaldal.com/",
    "https://www.rokomari.com/",
    "https://www.pikaboo.com/",
    "https://www.evaly.com.bd/"
]

OUT_PATH = "data/raw/web.jsonl"

def get_domain(url):
    return urlparse(url).netloc

def is_same_site(url, root_url):
    return get_domain(url).endswith(get_domain(root_url))

def extract_links(html, base_url):
    soup = BeautifulSoup(html, "lxml")
    links = set()

    for tag in soup.find_all("a", href=True):
        link = urljoin(base_url, tag["href"].split("#")[0])

        if link.startswith("http://") or link.startswith("https://"):
            links.add(link)

    return links

def crawl_domain(root_url, session, out_file, robot_parser):
    queue = [root_url]
    visited = set()
    saved = 0

    progress = tqdm(
        total=MAX_PAGES_PER_DOMAIN,
        desc=f"Crawling {get_domain(root_url)}"
    )

    while queue and saved < MAX_PAGES_PER_DOMAIN:
        url = queue.pop(0)

        if url in visited:
            continue

        visited.add(url)

        if not robot_parser.can_fetch("*", url):
            continue

        try:
            response = session.get(url, timeout=20)

            if response.status_code != 200:
                continue

            content_type = response.headers.get("content-type", "")

            if "text/html" not in content_type:
                continue

            text = trafilatura.extract(
                response.text,
                include_comments=False,
                favor_recall=True
            )

            if not text or len(text.split()) < MIN_WORDS:
                continue

            soup = BeautifulSoup(response.text, "lxml")
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else ""

            record = {
                "source": get_domain(root_url),
                "source_type": "web",
                "url": url,
                "title": title,
                "text": text,
                "crawl_date": time.strftime("%Y-%m-%d"),
                "language": "bn_or_en"
            }

            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved += 1
            progress.update(1)

            for link in extract_links(response.text, url):
                if link not in visited and is_same_site(link, root_url):
                    queue.append(link)

            time.sleep(DELAY_SECONDS)

        except Exception:
            continue

    progress.close()
    return saved

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    total_saved = 0

    with open(OUT_PATH, "w", encoding="utf-8") as out_file:
        for root_url in SOURCES:
            robot_parser = RobotFileParser()
            robot_parser.set_url(urljoin(root_url, "/robots.txt"))

            try:
                robot_parser.read()
            except Exception:
                pass

            saved = crawl_domain(root_url, session, out_file, robot_parser)
            total_saved += saved

            print(f"{root_url}: saved {saved} pages")

    print(f"Total web pages saved: {total_saved}")
    print(f"Saved web corpus to {OUT_PATH}")

if __name__ == "__main__":
    main()
```

Run it:

```bash
python 02_crawl_web.py
```

## 6. Save the corpus builder

Create a file named `03_build_corpus.py` and paste this code:

```python
import argparse
import hashlib
import json
import os
import re

from tqdm import tqdm

def clean_text(text):
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(text, min_words=80, max_words=300, overlap_sentences=1):
    sentences = re.split(r"(?<=[।!?\.])\s+", text)

    chunks = []
    current = []
    current_words = 0

    for sentence in sentences:
        words = len(sentence.split())

        if current_words + words > max_words and current:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_words = sum(len(x.split()) for x in current)

        current.append(sentence)
        current_words += words

        if current_words >= min_words:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_words = sum(len(x.split()) for x in current)

    if current and sum(len(x.split()) for x in current) >= min_words:
        chunks.append(" ".join(current))

    return chunks

def detect_domain(source, url):
    combined = f"{source} {url}".lower()

    if "dghs" in combined:
        return "health"

    if "moedu" in combined:
        return "education"

    if "dae" in combined:
        return "agriculture"

    if "gov.bd" in combined:
        return "government_civics"

    if ".ac.bd" in combined or "university" in combined:
        return "university_admissions"

    news_domains = [
        "prothomalo",
        "kalerkantho",
        "ittefaq",
        "jugantor",
        "samakal",
        "tbsnews",
        "thedailystar",
        "dhakatribune"
    ]

    if any(domain in combined for domain in news_domains):
        return "news_current_affairs"

    ecommerce_domains = [
        "daraz",
        "chaldal",
        "rokomari",
        "pikaboo",
        "evaly"
    ]

    if any(domain in combined for domain in ecommerce_domains):
        return "ecommerce_products"

    return "general_knowledge"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wikipedia", default="data/raw/wikipedia.jsonl")
    parser.add_argument("--web", default="data/raw/web.jsonl")
    parser.add_argument("--out", default="data/corpus.jsonl")
    parser.add_argument("--target", type=int, default=100000)
    parser.add_argument("--min-words", type=int, default=80)
    parser.add_argument("--max-words", type=int, default=300)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    seen_hashes = set()
    doc_id = 0
    total_passages = 0

    input_files = [args.wikipedia, args.web]

    with open(args.out, "w", encoding="utf-8") as out_file:
        for input_path in input_files:
            if not os.path.exists(input_path):
                print(f"Warning: file not found: {input_path}")
                continue

            with open(input_path, encoding="utf-8") as input_file:
                for line in tqdm(input_file, desc=f"Processing {input_path}"):
                    if total_passages >= args.target:
                        break

                    try:
                        record = json.loads(line)
                    except Exception:
                        continue

                    text = clean_text(record.get("text", ""))

                    if len(text.split()) < args.min_words:
                        continue

                    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()

                    if text_hash in seen_hashes:
                        continue

                    seen_hashes.add(text_hash)

                    source = record.get("source", "")
                    url = record.get("url", "")
                    domain = detect_domain(source, url)

                    chunks = chunk_text(
                        text,
                        min_words=args.min_words,
                        max_words=args.max_words
                    )

                    for chunk_id, chunk in enumerate(chunks):
                        if total_passages >= args.target:
                            break

                        doc_id += 1
                        total_passages += 1

                        passage = {
                            "doc_id": f"d_{doc_id:07d}",
                            "source": source,
                            "source_type": record.get("source_type", "wikipedia"),
                            "source_url": url,
                            "title": record.get("title", ""),
                            "domain": domain,
                            "chunk_id": chunk_id,
                            "text": chunk,
                            "language": record.get("language", "bn")
                        }

                        out_file.write(json.dumps(passage, ensure_ascii=False) + "\n")

    print(f"Created {total_passages} passages in {args.out}")

if __name__ == "__main__":
    main()
```

Run it:

```bash
python 03_build_corpus.py \
  --wikipedia data/raw/wikipedia.jsonl \
  --web data/raw/web.jsonl \
  --out data/corpus.jsonl \
  --target 100000
```

## 7. Check the corpus

Create a file named `check_corpus.py` and paste this code:

```python
import json
from collections import Counter

corpus_path = "data/corpus.jsonl"

passages = []

with open(corpus_path, encoding="utf-8") as file:
    for line in file:
        passages.append(json.loads(line))

doc_ids = [p["doc_id"] for p in passages]
texts = [p["text"] for p in passages]
domains = [p["domain"] for p in passages]

print("Total passages:", len(passages))
print("Unique doc IDs:", len(set(doc_ids)))
print("Unique passage texts:", len(set(texts)))
print("Duplicate doc IDs:", len(doc_ids) - len(set(doc_ids)))
print("Duplicate texts:", len(texts) - len(set(texts)))
print("Average words per passage:", sum(len(t.split()) for t in texts) / len(passages))
print("Domain distribution:")

for domain, count in Counter(domains).most_common():
    print(f"- {domain}: {count}")
```

Run it:

```bash
python check_corpus.py
```

## 8. Final output

After successful completion, your main corpus file will be:

```text
data/corpus.jsonl
```

It should contain **100,000 passages**, with each line in this format:

```json
{
  "doc_id": "d_0000001",
  "source": "bengali_wikipedia",
  "source_type": "wikipedia",
  "source_url": "https://bn.wikipedia.org/wiki/Example",
  "title": "Example title",
  "domain": "health",
  "chunk_id": 0,
  "text": "Clean passage text...",
  "language": "bn"
}
```

## Important: pilot first

Before running the full crawl, test the pipeline with a smaller target:

```bash
python 03_build_corpus.py \
  --wikipedia data/raw/wikipedia.jsonl \
  --web data/raw/web.jsonl \
  --out data/corpus_pilot.jsonl \
  --target 10000
```

Then inspect `data/corpus_pilot.jsonl`. If the passages look clean and correctly labeled, run the full 100,000-passage build.

The Wikipedia downloader uses Hugging Face’s `load_dataset()`, which downloads and caches the requested Parquet dataset locally; Trafilatura’s `extract()` returns the main readable content from a downloaded webpage. [trafilatura.readthedocs](https://trafilatura.readthedocs.io/en/stable/faq.html)