# Progress summary for the advisor (2026-11-14, before the 18 Nov pre-defence)

**Project:** A Benchmark and Hybrid Retrieval Framework for Bengali Code-Mixed Queries in RAG. Author: Montaha Tahsin (solo).

## What is done
- **Corpus:** 1,105,178 Wikipedia passages (Bengali 424,910, English 680,268), snapshot 2023-11-01.
- **Dataset audit:** I checked the 12,831-query pilot set against your ACL assessment. Your points hold: no training split, YouTube at 69.9%, source and domain confounded (all MIRACL and 93% of synthetic queries are "general"; all 391 unassigned are Facebook), mixed script only 8.2%, 16 duplicate groups and 42 near-duplicate pairs across dev and test.
- **Rebuilt splits:** grouped so near-copies stay together: train 8,136, dev 1,155, test 2,328, synthetic 1,209 kept separate.
- **Elicitation pilot:** 3 native writers wrote 60 entries (57 unique). A second native speaker judged 46 of 56 unique queries as ones a real person could ask (82%; one rater, no agreement measured).
- **Two pilot experiments** on 1,212 synthetic queries with known answer passages:
  - BM25 Recall@10: Bangla script 0.65, mixed 0.36, Romanized 0.03.
  - A training-free, corpus-derived normaliser lifts Romanized Recall@10 to 0.18 (paired nDCG@10 gain +0.096, 95% CI +0.073 to +0.120). Gain on mixed queries is small; Bangla unchanged.
- **Report:** pre-defence thesis (chapters 1 to 5) in the Stamford Word format.

## Limits (stated in the report)
- Only about 530 real queries have an answer passage in the corpus, nearly all Bangla script. Their labels come from BM25 plus a reranker, so their Recall@10 is not a finding.
- Synthetic queries are LLM-written and not yet checked by native speakers.
- Mined YouTube and Facebook text cannot be released (platform terms; YouTube data deleted by 2026-12-08). The public benchmark must use elicited or rewritten queries.
- No human relevance judgments, no dense or hybrid systems, no RAG study yet.

## Questions for you
1. **Scope:** your ACL method is domain classification across scripts; my benchmark is for retrieval. Should the thesis be the retrieval benchmark?
2. **Release:** do you agree the released set uses elicited or rewritten queries only?
3. **Timeline:** what is the thesis submission date, and may the defence be scoped to the pilot benchmark plus baselines (BM25, dense, fusion, normaliser), with human annotation and RAG as future work?
4. **Ethics:** is the ethics sign-off needed before collecting more elicited queries?
5. **Pre-defence:** are slides, a short report, or both expected?

## Next (after 18 Nov)
Dense baselines (multilingual E5, BGE-M3) and fusion; language-level tagger and code-mixing measures; collect more elicited and rewritten queries; pooled human annotation with agreement.
