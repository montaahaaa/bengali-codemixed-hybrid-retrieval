# Pilot corpus card

- **Source:** Hugging Face `wikimedia/wikipedia`, Bengali (`20231101.bn`) and English (`20231101.en`). `20231101` was the only snapshot available for both languages.
- **Licence:** Wikipedia text is CC BY-SA 4.0 (and GFDL). Reuse needs attribution and the same licence. This limits what can be released (tech-stack §11).
- **Downloaded:** 2026-11-03.
- **Bengali:** all articles (143,069) -> 424,910 passages.
- **English:** topic-filtered with `en_filter.json` (155,758 of 6,407,814 articles kept, 2.4%) -> 680,268 passages. Filter rules are in `en_filter.json`; counts per domain are in `en_filter_counts.json`. An article can count in more than one domain.
- **Passages:** whole sentences grouped up to 150 words (`scripts/chunking.py`). IDs are `<lang>-<article_id>-<n>`.
- **Totals:** 1,105,178 passages. Stats are in `stats.json`.

## Wikivoyage (separate file)
- **File:** `corpus_voyage.jsonl`, built by `scripts/build_voyage.py` (untracked in git, rebuildable). Not merged into `corpus.jsonl`.
- **Source:** en.wikivoyage.org and bn.wikivoyage.org via the MediaWiki API, fetched 2026-11-03/04. Licence CC BY-SA 4.0; each passage keeps `source`, `licence`, `revision` and `snapshot` (page timestamp).
- **Content:** 130 Bangladesh-related pages (26 en, 104 bn) -> 1,267 passages (572 en, 695 bn), same chunker as Wikipedia. IDs are `<lang>-wv<pageid>-<n>`.
- **Limits:** pages with leftover wikitext were dropped; Bengali pages were chosen by search for "বাংলাদেশ" and include some about neighbouring regions. Travel-oriented, so it adds little on fees.

## Known limits
- The English side is filtered by keyword and is not a random sample, so distractors lean towards the six domains.
- A probe of 30 expected titles found 21 kept. Seven general articles that should match by title (Passport, Dengue fever, Vaccine, Insurance, Citizenship, Password, Airport) were not in the kept set. They were not in the first 15 of 41 English shards either. It is unresolved whether they are missing from the dataset or the filter missed them. Phase 1.5 shows whether this matters.
- 24.0% of Bengali and 15.9% of English passages are shorter than 100 words.
- `corpus.jsonl` is not tracked in git. It is rebuilt with `scripts/build_corpus.py`.

## corpus_plus.jsonl (optional combined file, 2026-11-15)
- **Contents:** pilot Wikipedia (bn, en) + Bengali Wikibooks + Wikivoyage (en, bn). Pilot `corpus.jsonl` is unchanged and is still the corpus behind every reported number.
- **Wikibooks cleaning:** dropped passages with leftover MediaWiki magic words (`__NOTOC__`), the main page and passages under 40 words.
- **Not added:** Wikisource bn (345 MB dump, literature, adds distractors not answers), Wikiquote, Wiktionary, Wikinews (no bn dump); WHO web pages (all rights reserved); NHS (OGL but English, health only, reuse conditions); mC4/OSCAR/CulturaX/CC-100 (web crawls, rights to the underlying text unclear, not releasable under D-030).
- **Not indexed:** BM25 and dense indexes still use `corpus.jsonl`. Re-indexing changes every score, so do it after the pre-defence.
