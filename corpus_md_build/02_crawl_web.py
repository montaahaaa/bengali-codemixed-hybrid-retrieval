import json
import os
import time
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
import trafilatura
from tqdm import tqdm

USER_AGENT = "BengaliCodeMixedRAGBenchmark/1.0 (academic research; non-commercial)"
DELAY_SECONDS = 1.5
MAX_PAGES_PER_DOMAIN = 1000
MIN_WORDS = 80

SOURCES = [
    "https://www.dghs.gov.bd/",
    "https://mohfw.gov.bd/",
    "https://moa.gov.bd/",
    "https://mopme.gov.bd/",
    "https://dshe.gov.bd/",
    "https://bteb.gov.bd/",
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

    requests_made = 0
    while queue and saved < MAX_PAGES_PER_DOMAIN and requests_made < MAX_PAGES_PER_DOMAIN * 4:
        requests_made += 1
        url = queue.pop(0)

        if url in visited:
            continue

        visited.add(url)

        if not robot_parser.can_fetch("*", url):
            continue

        try:
            response = session.get(url, timeout=20)
            time.sleep(DELAY_SECONDS)

            if response.status_code != 200:
                continue

            content_type = response.headers.get("content-type", "")

            if "text/html" not in content_type:
                continue

            for link in extract_links(response.text, url):
                if link not in visited and is_same_site(link, root_url):
                    queue.append(link)

            text = trafilatura.extract(
                response.text,
                include_comments=False,
                favor_recall=True
            )

            if not text or len(text.split()) < MIN_WORDS:
                continue
            lines = [x for x in text.split("\n") if x.strip()]
            if lines and sum(1 for x in lines if x.lstrip().startswith("|")) / len(lines) > 0.3:
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

            out_file.write(json.dumps(record, ensure_ascii=False) + "\n"); out_file.flush()
            saved += 1
            progress.update(1)


        except Exception:
            continue

    progress.close()
    return saved

def main():
    global MAX_PAGES_PER_DOMAIN, SOURCES, OUT_PATH
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--max-pages",type=int,default=MAX_PAGES_PER_DOMAIN)
    ap.add_argument("--only",nargs="*",help="substrings of source URLs to crawl")
    ap.add_argument("--out",default=OUT_PATH)
    a=ap.parse_args()
    MAX_PAGES_PER_DOMAIN=a.max_pages; OUT_PATH=a.out
    if a.only: SOURCES=[u for u in SOURCES if any(o in u for o in a.only)]
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
