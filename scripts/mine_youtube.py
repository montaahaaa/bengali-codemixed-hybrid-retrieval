"""Mine a small pilot sample of YouTube comments through the official YouTube Data API v3 (D-033).
Usage: python scripts/mine_youtube.py [--target 500] [--videos-per-seed 3]
Needs YOUTUBE_API_KEY in .env (never commit it). No scraping outside the API.
Rules from data/annotations/pilot/platform_terms.md:
  - pilot labelling only; do NOT release comment text; delete or refresh within 30 days
    (every row carries fetched_at and delete_after);
  - author names and ids are not stored.
Quota: search.list = 100 units, commentThreads.list = 1 unit; default daily quota 10,000.
Output: data/raw/youtube/comments.jsonl (git-ignored) and data/raw/youtube/fetch_log.json."""
import argparse, json, os, sys, time
import urllib.error, urllib.parse, urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data/raw/youtube"
API = "https://www.googleapis.com/youtube/v3"

# Search seeds per domain (Bangla / Romanized / English), derived from data/annotations/pilot/topics.md.
SEEDS = {
    "fees": ["পাসপোর্ট ফি কত", "ইনকাম ট্যাক্স রিটার্ন জমা", "bkash cash out charge", "ব্যাংক একাউন্ট খোলার নিয়ম"],
    "procedures": ["এনআইডি কার্ড সংশোধন", "জন্ম নিবন্ধন অনলাইন", "ড্রাইভিং লাইসেন্স করার নিয়ম", "জমির নামজারি"],
    "health": ["ডেঙ্গু প্রতিরোধ", "ডায়াবেটিস নিয়ন্ত্রণ", "শিশুর টিকা", "জ্বর হলে কী করবেন"],
    "education": ["এইচএসসির পর ভর্তি", "বিদেশে পড়াশোনা স্কলারশিপ", "বিসিএস প্রস্তুতি", "spoken english bangla"],
    "tech": ["ফেসবুক পাসওয়ার্ড ভুলে গেলে", "wifi slow solution bangla", "অনলাইন স্ক্যাম থেকে সাবধান", "freelancing শুরু করবেন কীভাবে"],
    "local_services": ["ট্রেনের টিকিট কাটার নিয়ম", "বিদ্যুৎ বিল পরিশোধ", "ঢাকা মেট্রোরেল", "জাতীয় জরুরি সেবা ৯৯৯"],
}


def load_key():
    key = os.environ.get("YOUTUBE_API_KEY", "")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("YOUTUBE_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"')
    if not key:
        sys.exit("YOUTUBE_API_KEY missing. Add it to .env (see .env.example).")
    return key


def call(endpoint, params, key):
    url = f"{API}/{endpoint}?" + urllib.parse.urlencode({**params, "key": key})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            full = e.read().decode("utf-8", "replace")
            body = full[:300]
            if endpoint == "commentThreads" and e.code in (403, 404) and "quota" not in full.lower():
                return None  # comments disabled, video removed or private
            if e.code == 403 and "quota" in full.lower():
                sys.exit(f"Quota exhausted: {body}")
            if e.code >= 500 and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            sys.exit(f"HTTP {e.code} on {endpoint}: {body}")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--videos-per-seed", type=int, default=3)
    ap.add_argument("--per-video", type=int, default=100, help="max comments kept per video (spreads the sample over channels)")
    ap.add_argument("--reuse-log", help="fetch_log.json from an earlier run: skip search.list, take more comments from those videos")
    args = ap.parse_args()
    old = {}
    if args.reuse_log:
        for e in json.load(open(args.reuse_log, encoding="utf-8"))["searches"]:
            old[(e["domain"], e["seed"])] = e["videos"]
    key = load_key()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "comments.jsonl"
    seen = set()
    if out_path.exists():
        seen = {json.loads(l)["comment_id"] for l in out_path.open(encoding="utf-8")}
    now = datetime.now(timezone.utc)
    stamp, expiry = now.isoformat(timespec="seconds"), (now + timedelta(days=30)).date().isoformat()
    units, n_new, videos_done = 0, 0, set()
    log = {"fetched_at": stamp, "delete_after": expiry, "searches": [], "quota_units_est": 0}

    with out_path.open("a", encoding="utf-8") as f:
        # Round-robin over domains so the sample stays balanced if the target is reached early.
        rounds = max(len(v) for v in SEEDS.values())
        per_domain_target = args.target // len(SEEDS) + 1
        counts = {d: 0 for d in SEEDS}
        for i in range(rounds):
            for domain, seeds in SEEDS.items():
                if i >= len(seeds) or counts[domain] >= per_domain_target:
                    continue
                q = seeds[i]
                if args.reuse_log:
                    vids = old.get((domain, q), [])
                else:
                    res = call("search", {"part": "snippet", "q": q, "type": "video", "maxResults": args.videos_per_seed,
                                          "relevanceLanguage": "bn", "regionCode": "BD"}, key)
                    units += 100
                    vids = [it["id"]["videoId"] for it in (res or {}).get("items", [])]
                log["searches"].append({"domain": domain, "seed": q, "videos": vids})
                for vid in vids:
                    if vid in videos_done or counts[domain] >= per_domain_target:
                        continue
                    videos_done.add(vid)
                    token, got = None, 0
                    while counts[domain] < per_domain_target and got < args.per_video:
                        p = {"part": "snippet", "videoId": vid, "maxResults": min(100, args.per_video - got), "textFormat": "plainText", "order": "relevance"}
                        if token:
                            p["pageToken"] = token
                        data = call("commentThreads", p, key)
                        units += 1
                        if not data:
                            break
                        for it in data.get("items", []):
                            s = it["snippet"]["topLevelComment"]["snippet"]
                            cid = it["snippet"]["topLevelComment"]["id"]
                            if cid in seen or got >= args.per_video:
                                continue
                            seen.add(cid)
                            row = {"comment_id": cid, "video_id": vid, "domain": domain, "seed": q,
                                   "text": s["textDisplay"], "published_at": s["publishedAt"],
                                   "like_count": s.get("likeCount", 0), "fetched_at": stamp, "delete_after": expiry}
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                            counts[domain] += 1
                            got += 1
                            n_new += 1
                        token = data.get("nextPageToken")
                        if not token:
                            break
                if n_new >= args.target:
                    break
            if n_new >= args.target:
                break
    log.update(new_comments=n_new, per_domain=counts, quota_units_est=units, videos=len(videos_done))
    (OUT_DIR / "fetch_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {n_new} new comments from {len(videos_done)} videos (~{units} quota units). Delete after {expiry}.")
    print("Per domain:", counts)


if __name__ == "__main__":
    main()
