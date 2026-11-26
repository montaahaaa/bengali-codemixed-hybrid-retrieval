# A Benchmark for Bangla Passage Retrieval with Bangla-Script, Code-Mixed and Romanized Queries

A passage-retrieval benchmark for Bangla questions written in three ways: **Bangla script**, **mixed Bangla-English**, and **Romanized Bangla** (Bangla typed in English letters).

A system gets a question and must find the Wikipedia passage that answers it.

## Files

| File | Rows | Columns |
|---|---|---|
| `queries.csv` | 10,264 | `query_id, query, script, domain, split` |
| `qrels.csv` | 10,264 | `query_id, passage_id, relevance` |
| `corpus.csv` | 99,567 | `passage_id, title, text, source_url` |

- `script` is `bangla`, `mixed` or `romanized`.
- `split` is `train` (7,246), `dev` (1,564) or `test` (1,454). The split is done by passage, so all questions about one passage are in the same split.
- `relevance` is always 1: each question has one known answer passage.
- Final build. Rebuild with `corpus_md_build/26_build_submission.py`.

## How the questions were made
1. **Corpus.** Bengali Wikipedia (snapshot 20250620), cut on sentence boundaries into passages targeting 80 to 300 words (99.9% are in that range; mean 91 words; a few single long sentences give extremes of 1 to 1,165 words).
2. **Questions.** A language model read one passage at a time and wrote a question meant to be answerable from that passage alone (not always achieved, see Quality checks), then a second call checked "does this passage answer this question?" and kept only the "yes" cases.
3. **Two query groups, told apart by `query_id`:**
   - IDs starting with `P` (3,441 queries, 1,147 passages, each with all three forms) are the **parallel** part: the Bangla, mixed and Romanized versions of a passage's question ask the *same* thing. IDs end in `_b`, `_m`, `_r`.
   - IDs starting with `G` (6,823 queries) are the **non-parallel** part (v1): the three versions of a passage usually ask *different* facts. Use the `P` part for any claim about the effect of script.
4. **Filters.** Script checked from the characters, queries under 3 words and exact duplicates removed, one near-duplicate removed, government-site passages removed.

Generators: the `G` queries by `claude-haiku-5-5`; the `P` queries by `claude-haiku-5-5` (825 passages) and `openai/gpt-oss-120b` via Groq (322 passages).

## Baselines (nDCG@10, parallel `P` test split, 498 queries)
| Script | BM25 | Dense (multilingual-E5-base) | Hybrid (BM25 + dense, convex weight 0.7) |
|---|---|---|---|
| Bangla | 0.58 | 0.71 | 0.74 |
| Mixed | 0.34 | 0.66 | 0.63 |
| Romanized | 0.01 | 0.12 | 0.07 |
| All | 0.31 | 0.50 | 0.48 |

Main finding: BM25 almost fails on Romanized queries; a Romanized normaliser raises hybrid nDCG@10 on Romanized queries by +0.15 (95% CI [+0.12, +0.18], dev and test combined).

## Quality checks
- 170 `G` queries (100 Romanized, 35 Bangla, 35 mixed) were reviewed earlier by one native speaker; 54 of the Romanized texts were corrected in that review. None of them is in the `P` part.
- A random sample of 34 parallel passages (102 queries, spread over 13 domains) was reviewed by one native speaker (single rater, so no agreement figure). Results, as yes-rates over the 34 queries per script:

| Check | Bangla | Mixed | Romanized |
|---|---|---|---|
| Natural | 85% | 71% | 79% |
| Answerable from the passage | 97% | 97% | 97% |
| Same meaning as the Bangla form | reference | 100% | 100% |

- The reviewer found that about 9 of 34 passages (26%) have a question that does not name what it asks about (for example "this school", "his post"), so several passages could match. Of the reviewer's 25 written corrections, 24 are applied in `queries.csv` (the other would have changed a mixed query into pure Bangla and was left as is). The corrected queries are the P-IDs of the reviewed sample (`corpus_md_build/data/review_overrides.json`).
- For the rest, a simple rule lists queries with pointer words ("এই…", "তার…") for a second look: 1,161 of the 10,162 queries outside the 34-passage sample (296 demonstratives, 865 pronouns). The rule found only 4 of the reviewer's 9 under-specified passages (recall 0.44, precision 0.57), so it is a list to inspect, not a filter; **no query was removed**. The list is `review/flagged_queries.csv` (not part of the dataset files).
- All other queries are **unreviewed model output**.

## Licence
- Passage text: **CC BY-SA 4.0** (Wikipedia). Keep `title` and `source_url` for attribution; adaptations must be shared under the same licence.
- Queries and qrels: model-generated. *(State your chosen licence and check the generator providers' terms before publishing.)*
- Government-site passages and their queries are **not included** (no open licence).
- No personal data is collected. The corpus is Wikipedia text, so it can mention public figures and real people; queries are model-written from it.

## Known limits
- **One answer passage per question.** Other passages may also answer it, so scores are lower bounds.
- **Some questions are under-specified.** In the reviewed sample about a quarter did not name their subject, so the gold passage may not be the only plausible match; the same is likely in the unreviewed part (see Quality checks). Baselines were run before the 24 corrections and are not rerun (24 of 10,264 queries).
- **Mostly unreviewed, model-written questions.** They are cleaner and more fact-seeking than real user questions. Romanized spelling follows model habits.
- **The `G` part is not parallel**, see above.
- **Topic skew.** Encyclopedic content; about a third is history and culture, little everyday practical text.
- **Same-family check.** The generator and the answerability check are the same model family.
- **Lexical overlap.** A no-copy rule limited word overlap between question and passage. Measured on the `G` part, the mean share of query words found in the gold passage is 0.40 (Bangla), 0.26 (mixed) and 0.005 (Romanized), which favours lexical systems on Bangla and mixed queries. It was not measured for the `P` part.
- **Distractors.** Only 3,546 of the 99,567 passages are the answer to any query; the rest are distractors with no queries.
- **Small corpus** (about 100,000 passages): absolute scores are higher than they would be on a larger collection.

## Reproduce
Scripts are in `corpus_md_build/`: `25_clean_parallel.py` (clean), `22_build_v2.py` (parallel set), `26_build_submission.py` (these CSVs), `14v2_baselines.py`, `16v2_normaliser.py`, `18v2_adaptive_fusion.py` (baselines).
