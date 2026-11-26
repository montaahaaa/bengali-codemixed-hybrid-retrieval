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
    parser.add_argument("--max-web-share", type=float, default=0.6)
    parser.add_argument("--priority", default=None, help="JSON list of Wikipedia page ids to include first")
    parser.add_argument("--min-words", type=int, default=80)
    parser.add_argument("--max-words", type=int, default=300)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    seen_hashes = set()
    doc_id = 0
    total_passages = 0

    priority = set(json.load(open(args.priority))) if args.priority else set()
    input_files = ([args.wikipedia] if priority else []) + [args.web, args.wikipedia]
    first_pass = bool(priority)
    web_cap = int(args.target * args.max_web_share)

    with open(args.out, "w", encoding="utf-8") as out_file:
        for pass_no, input_path in enumerate(input_files):
            if not os.path.exists(input_path):
                print(f"Warning: file not found: {input_path}")
                continue

            with open(input_path, encoding="utf-8") as input_file:
                for line in tqdm(input_file, desc=f"Processing {input_path}"):
                    if total_passages >= args.target:
                        break
                    if input_path == args.web and total_passages >= web_cap:
                        break

                    try:
                        record = json.loads(line)
                    except Exception:
                        continue

                    if first_pass and pass_no == 0:
                        m = re.search(r"curid=(\d+)", record.get("url", ""))
                        if not m or m.group(1) not in priority:
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
