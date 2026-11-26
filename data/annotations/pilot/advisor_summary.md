# Pilot query dataset: summary for advisor (2026-11-09)

**Project:** Bangla and code-mixed (Bangla–English, Romanized) retrieval benchmark, Bangladesh context. This is the Stage 2 pilot on how to source queries.

## What was built
- **Mining dataset: 10,000 candidate queries** (`dataset_10000.csv`): 9,391 YouTube comments and 609 Facebook queries.
  - YouTube: official YouTube Data API v3, 227 videos found by 24 Bangla/Romanized/English seed searches over six domains (fees, procedures, health, education, tech, local services). 28,711 raw comments were collected, 25,869 remained after cleaning and de-duplication.
  - Facebook: 609 queries from a hand-collected sheet (D-034); comments and grades were deleted, queries only.
  - A rough rule (question-like, at least 4 words, no praise, link or phone number) kept 11,771 items. 10,000 were selected, balanced across the six YouTube domains as far as supply allowed.
- **Script mix:** 6,386 Bangla script, 2,795 Romanized/Latin, 819 mixed.
- **Corpus:** 1.1M passages from Bengali and English Wikipedia (2023-11-01 snapshot), CC BY-SA 4.0.

## How each item was graded
Every item was searched against the corpus with BM25 (top 10), then the 10 candidates were reranked with a multilingual cross-encoder (BAAI/bge-reranker-v2-m3). Grade 2 = a passage likely answers it (score >= 0.60), 1 = topical (>= 0.20), 0 = otherwise.

| | Items | Grade 0 | Grade 1 | Grade 2 |
|---|---|---|---|---|
| Owner-checked grades | 5,609 | 5,242 | 262 | 105 |
| Not yet checked (auto only) | 4,391 | 4,151 | 188 | 52 |
| **Total** | **10,000** | **9,393** | **450** | **157** |

**Answerable yield: 157 of 10,000 (1.6%).** Against all 25,869 cleaned raw items it is about 0.6%.

## Draft benchmark (`benchmark_draft/`)
- 107 queries: 32 dev, 75 test (seed 42), 72 distinct answer passages, relevance judgments in TREC qrels format (grade 2).
- Built from the 157 grade-2 items after a model-drafted review (keep 21, weak 86, drop 50), with a clean Bangla rewrite of each query. **All of this review and the rewrites are machine drafts awaiting manual checking.**
- Domain skew: health 63, fees 15, procedures 12, local services 8, tech 4, education 3.

## Limits (please read)
1. **Grades are a proxy.** The owner confirmed the 5,609 first-batch grades; the 4,391 new ones are unchecked. Zero-grade items were not individually re-read, so recall is unmeasured.
2. **Romanized queries are nearly all grade 0** (BM25 cannot reach Bangla passages from Latin script). This is a retrieval limit, not proof of no answer. A dense multilingual retriever would be needed to measure it.
3. **Most real needs are not answerable from Wikipedia** (local, time-sensitive, personal, or fee/procedure details). Government and service pages (kept private, D-029) would raise fee and procedure coverage.
4. **Most health items are long personal cases** whose passage is only topical.
5. **Release limits.** YouTube text cannot be released and must be deleted by 2026-12-08 (API terms, D-033). Facebook text cannot be released (D-033). A releasable benchmark must use elicited or rewritten queries (roadmap 2.4, 2.5). Clean rewrites here are drafts derived from comments and are not release-cleared.
6. Source terms: Facebook groups dropped as an automated route; YouTube used only through the official API as a private pilot; Reddit terms unverified.

## Open decisions for the advisor
- Which sourcing route to adopt (mining, elicitation, rewriting): mining's low yield and unreleasable text argue for elicited or rewritten queries. Elicitation sheets for 3 native speakers are prepared (`elicitation/`).
- Whether to add government-service text to the corpus (D-029) and use a dense retriever for Romanized queries.
- Whether to apply for Meta Content Library access (needs institutional approval).
