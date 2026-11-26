# Data card: Bengali code-mixed retrieval benchmark (pilot v1)

Status: **pilot, mostly unreviewed.** Built 2026-11-16. Part of the thesis *A Benchmark and Hybrid Retrieval Framework for Bengali Code-Mixed Queries in RAG*. Decisions: `specs/decision.md` D-046 to D-050.

## What it is
A passage-retrieval test collection in BEIR format: a 100,000-passage corpus, 10,162 queries in three script forms (Bangla script, mixed Bangla-English, Romanized Bangla), and one relevant passage per query.

## Files (`corpus_md_build/data/`)
| File | Content |
|---|---|
| `corpus.jsonl` | 100,000 passages: `doc_id`, `source`, `source_type`, `source_url`, `title`, `domain`, `chunk_id`, `text`, `language` |
| `queries.jsonl` | 10,162 queries: `_id`, `text`, `metadata` (script, domain, split, source tier, source_doc, review, text_original when corrected) |
| `qrels.csv` | `query-id, corpus-id, score` (score 2 = the passage the query was written from) |
| `review/` | 145-need review sheet, instructions, and the Romanized review results |
| `baseline_results.json`, `normaliser_results.json`, `reviewed_rerun_results.json`, `adaptive_fusion_results.json` | Reported baselines |
| `backup_pre_finalize/` | Queries and qrels before the review corrections and near-duplicate pass |

## How it was built
1. **Corpus.** Bengali Wikipedia (`sagorsarker/bangla-wikipedia`, snapshot 20250620): 99,567 passages of 80 to 300 words, cut on sentence boundaries. First every article that had been an answer page in earlier pilot work, then random fill. Plus 433 passages crawled from Bangladesh government sites (`mopa.gov.bd`, `moedu.gov.bd`, others), after a robots.txt check and a table/form noise filter.
2. **Queries.** `claude-haiku-5-5` read 3,599 passages (70 to 160 words, picked by topic keywords, balanced across eight domain groups) and wrote three questions per passage: Bangla, Romanized and code-mixed, each answerable only from that passage, with no more than two consecutive words copied. A second call per query asked whether the passage answers it (yes/no); only "yes" was kept. The model also labelled each passage's domain.
3. **Filters.** Dropped 195 queries whose script did not match the label (checked from characters), queries under 3 words, exact duplicates, and one near-duplicate (word Jaccard of 0.7 or more within a script).
4. **Splits.** Hash of the source passage: train 70%, dev 15%, test 15%. All queries from one passage stay together.
5. **Review (partial).** A native speaker reviewed the Romanized query of 145 needs (sampled across domains): 64 kept as written, 81 corrected (mostly small spelling fixes; 76 of 81 are at least 95% similar to the original). Corrected text replaced the original; the original is in `metadata.text_original`.

## Composition
| | Bangla | Mixed | Romanized | Total |
|---|---|---|---|---|
| Train | 2,462 | 2,268 | 2,453 | 7,183 |
| Dev | 527 | 491 | 525 | 1,543 |
| Test | 493 | 453 | 490 | 1,436 |
| **All** | **3,482** | **3,212** | **3,468** | **10,162** |

Average query length: 10.2 words. Gold passages: 3,560 distinct. Domains (generator-labelled): history_culture 3,458; technology 1,061; people 977; health 886; government_law 785; economy_fees 614; other 519; science 518; education 505; geography 414; transport_local 345; agriculture 71; sports 9.

Quality tiers: 9,727 queries are `llm_generated_unreviewed`; 435 (145 needs x 3 forms) are `llm_generated_human_reviewed` by one native speaker. In the reviewed core, all 145 Bangla and 145 mixed queries were marked natural and answerable; of the Romanized ones, 81 were corrected and 64 kept as written, and one (R005, unclear wording) was marked not answerable and is flagged `exclude_from_core`. Seven needs carry reviewer notes (R005, R070, R105, R113, R118, R127, R138); R070, R118 and R127 point to possible ambiguity.

## Baselines (nDCG@10, test, 1,436 queries)
BM25 0.288; dense multilingual-E5-base 0.482; fixed-weight hybrid with normalised BM25 0.524; adaptive per-script fusion 0.541. By script (BM25 / dense / adaptive): Bangla 0.54 / 0.70 / 0.74; mixed 0.30 / 0.64 / 0.67; Romanized 0.02 / 0.13 / 0.22. No system was tuned on dev or test.

## Licences and release
- Wikipedia text: CC BY-SA 4.0 (attribution and share-alike); keep source URL and title.
- The 433 government-site passages (42 queries have one as their gold passage) carry **no open licence**. Do not release them; drop those passages and queries from any public version.
- Queries are model output (generator `claude-haiku-5-5`); check the provider's terms before a public release. Not legal advice.
- No personal data is intended: the corpus is encyclopedic text and the queries are written from it. Earlier mined YouTube/Facebook queries are **not** part of this dataset.

## Known limits and biases
- **The three script forms of a need are NOT parallel.** The generation prompt asked for three questions about *different facts when possible*, so the Bangla, mixed and Romanized queries of one passage usually ask different things (confirmed by reviewer 2: same meaning in 1 of 50 needs). Differences between scripts therefore mix the effect of script with the effect of the question. A parallel version (same question in three forms) is needed before claims about script effects (RQ1, RQ2) are made from this set.
- **Mostly unreviewed.** Only 145 needs (435 queries, 4% of the set) have a human check, by a single reviewer; the 50 rows marked for double review were rated once, so there is no agreement figure. Blanket "yes" on every Bangla and mixed row is unusual and should be spot-checked by a second person.
- **One gold passage per query.** Other passages may also answer a query, so recall and nDCG are lower bounds and systems that find other valid answers are penalised.
- **Topic skew.** Encyclopedic: about a third is history and culture; agriculture has 71 queries. The corpus has little practical text on fees, procedures and local services, so the dataset does not yet reflect everyday information needs.
- **LLM-written style.** Queries are cleaner and more fact-seeking than real search or chat text; Romanized spelling follows model habits (81 of 145 reviewed Romanized queries, 56%, needed a spelling or word-form fix; see `review/romanized_review_results.csv`). Real user queries were not used.
- **Same-family circularity.** The generator and the answerability check are the same model family.
- **Lexical overlap.** The no-copy rule limits it, but overlap with the gold passage was not measured, which can favour BM25.
- **Corpus size.** 100,000 passages is small; the answer articles were over-represented when the corpus was built, so absolute scores overstate retrieval from a larger collection.
- **Normaliser and gold passages share a corpus.** The Romanized normaliser lookup is built from the same corpus; the reviewed rerun suggests its gain does not depend on exact LLM spellings, but human-written Romanized queries have not been tested.

## Intended use
Pilot comparison of retrieval systems on Bangla, mixed and Romanized queries; tuning on train, reporting on dev/test. Not for claims about real user behaviour until a human-written and reviewed core is added.
