# Audit of `data/benchmark_full` against the advisor's assessment (2026-11-14)

The 12,831-query dataset the advisor reviewed is `data/benchmark_full` (the advisor saw only a description, not the CSV). Numbers below are measured from `queries.csv` and `qrels.tsv`.

| Advisor point | Status | Evidence |
|---|---|---|
| No train split | **Open** | Splits are dev 3,486 / test 8,133 / synthetic 1,212. No train. |
| Source imbalance | **Confirmed** | YouTube 8,970 (69.9%), MIRACL 2,042, synthetic 1,212, Facebook 607. |
| Source-domain confound | **Confirmed, worse than stated** | All 2,042 MIRACL queries are domain `general`. 1,122 of 1,212 synthetic are `general`. All 391 `unassigned` are Facebook. A domain classifier can learn the source. |
| Synthetic skew | **Confirmed** | 1,122 of 1,212 synthetic queries (93%) are `general`. |
| 391 unassigned | **Open** | All Facebook: exclude from domain-supervised runs or label. |
| Script imbalance | **Confirmed** | bn 8,529 (66.5%), latin 3,256 (25.4%), mixed 1,046 (8.2%). Mixed is thin; 269 of 1,046 are synthetic. |
| Leakage / duplicates | **Partly open** | 16 groups differ only by punctuation/spacing; 6 cross dev/test. 42 near-duplicate pairs (word Jaccard >= 0.8); 20 cross dev/test. Small but real. |
| Annotation reliability | **Open** | 5,371 owner-confirmed, 4,206 auto-unchecked, 1,212 LLM-unchecked, 2,042 published (MIRACL). No agreement study. Elicitation sheets for 3 writers exist, not collected. |
| Answerable yield | **Weak** | Only 104 `core` and 426 `partial` queries (about 530 of 12,831; 3,784 have any qrel row, mostly MIRACL). 9,006 are `unanswered`. |
| Romanized/code-mixed answerability | **Weak** | Of 3,256 Latin queries, 2 are `core`, 15 `partial`. BM25 cannot reach Bangla passages from Latin script (D-035). |
| Parallel cross-script subset | **Missing** | No query has linked native/romanized/code-mixed versions. Required for any script-invariance claim. |
| Platform terms | **Blocking for release** | YouTube text (8,970) cannot be released and must be deleted by 2026-12-08 (D-033). Facebook text cannot be released. Release needs elicited or rewritten queries. |
| Baselines / model | **Not started** | No experiments in the repo yet. |

## What this means
1. The paper's usable evidence today is small: about 530 answerable queries, nearly all Bangla script. The advisor's "12,831 queries" overstates what supports retrieval or script claims.
2. Real user queries cannot be released. The releasable benchmark must come from elicitation or rewriting (roadmap 2.4, 2.5), which fits building the parallel subset.
3. Splits must be rebuilt with grouping so equivalent and near-duplicate queries stay together, and with a source-held-out option.

## Suggested order
1. Rebuild splits (train/dev/test, grouped, dedup by normalized text plus similarity).
2. Collect the 3-writer elicitation sheets and use them as the seed of the parallel subset.
3. Decide scope with the advisor: classification (domain) vs retrieval. The advisor's loss and experiments are for domain classification; the repo's benchmark is a retrieval benchmark.
4. Run baselines.
