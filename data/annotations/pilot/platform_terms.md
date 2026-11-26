# Platform terms and access routes (phase 2.2)

Read on 2026-11-04 by fetching the official pages. Anything not read from a page is marked UNVERIFIED. Not legal advice. Raw platform text stays in `data/raw/` (git-ignored); only anonymised derived statistics are committed.

| Source | Permitted route found | Status |
|---|---|---|
| Reddit | Official Data API (expected) | **UNVERIFIED**: redditinc.com policy pages could not be fetched. Needs the project owner to read and paste the terms |
| YouTube comments | Official YouTube Data API v3 (`commentThreads`) | **Usable with limits** (below) |
| Facebook groups | Meta Content Library application only | **Dropped for now** (below) |
| Q&A forum / archive | Stack Exchange dump (CC BY-SA 4.0) | Licence OK, but **no Bengali site found**; Bengali Q&A source still open |

## YouTube (YouTube API Services Terms, Developer Policies)
- No research or academic exemption and no researcher programme appears on the pages read. The pages say the API is for operating an "API Client"; research use is not named, so using it for a pilot is an interpretation, not a stated permission.
- Needs an API key (one API project per client). Credentials must not be shared or embedded in open-source code, so keep the key in `.env` only.
- Scraping outside the API is not allowed (from the policy summary text; recheck the live page).
- Stored API data must not be kept indefinitely; the policy summary says refresh or delete after 30 days. Deletion on revocation within 7 or 30 days.
- ToS 16: YouTube keeps all rights in API data; nothing read grants a right to publish comment text.
- **Consequence:** collect a small sample into `data/raw/`, use it only for the pilot labelling, delete or refresh within 30 days, and do **not** put YouTube comment text in the released benchmark. Released queries must come from elicited or rewritten text. Commit only counts and yields.
- Not read: the revision history, and the Developer Policies text after the fetch cut-off (about 39k of 94k characters).

## Facebook groups (Meta Terms of Service, Platform Terms, Meta Content Library page, updated 2026-04-30)
- ToS 3.2.3: automated collection is not allowed without Meta's prior permission. No research exemption.
- Only route described: Meta Content Library, which covers public group posts. It needs an academic or not-for-profit affiliation and an independent review by CASD. No fee. Researchers may publish results without Meta's approval. Bengali coverage is not stated.
- The Groups API status was not checked (Developer Docs not read).
- **Decision:** drop Facebook groups from the pilot. Revisit only if the supervisor's institution applies for Content Library access.

## Q&A forum or archive
- Stack Exchange data dump (archive.org, dated 2024-04-02): CC BY-SA 4.0. Reuse requires attribution of the source, a link to each question, author names, and links to author profiles; share-alike applies. The site list read had no Bengali-language site (the list was cut off, so not fully checked).
- Wikimedia Reference desk and Bengali Wikipedia village pump: pages returned 404, UNVERIFIED.
- Other Bangla Q&A sites: not checked.
- **Consequence:** Stack Exchange is permitted but unlikely to hold Bangla or Romanized-Bangla queries. Candidate replacement: a permitted Bengali Q&A source still to be found.

## Open items for the project owner
1. Read Reddit's Data API Terms and User Agreement in a browser and paste the sections on research use, archives and publishing.
2. Confirm that the YouTube pilot use is acceptable for the thesis, or drop YouTube too.
3. Choose a replacement Bengali Q&A source.
