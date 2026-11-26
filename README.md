# Bengali Code-Mixed Passage Retrieval: Benchmark and Hybrid Retrieval

Bengali speakers write the same question in different ways: in Bangla script, mixed with English, or Romanized (Bangla typed in Latin letters). This repository builds a passage-retrieval benchmark that covers all three forms and measures how sparse, dense and hybrid retrievers cope with them.

| Form (`script` value) | Example: "In which year's NBA draft did the Lakers pick Magic Johnson?" |
|---|---|
| Bangla script (`bangla`) | লেকার্স ম্যাজিক জনসনকে কোন সালের এনবিএ ড্রাফটে বেছে নিয়েছিল? |
| Mixed (`mixed`) | Lakers ম্যাজিক জনসনকে কোন সালের NBA draft-এ select করেছিল? |
| Romanized (`romanized`) | Lakers Magic Johnsonke kon sheler NBA draft e beche niyechilo? |

## Status

| Part | State |
|---|---|
| Corpus and benchmark (10,264 queries, 99,567 passages) | Built |
| BM25, dense and hybrid baselines | Measured |
| Romanized skeleton normaliser | Measured |
| Fusion-weight selection (single and per-script) | Measured |
| Human review | Small sample only |
| Code-mixing measures, reranker, more dense models | Not done |
| Pooled, graded relevance judgments | Not done |
| End-to-end RAG evaluation | Not started |

[`research_proposal.md`](research_proposal.md) is the original plan. It describes intended work, and parts of it are not implemented.

## Benchmark

| File | Rows | Columns |
|---|---|---|
| [`submission/queries.csv`](submission/queries.csv) | 10,264 | `query_id, query, script, domain, split` |
| [`submission/qrels.csv`](submission/qrels.csv) | 10,264 | `query_id, passage_id, relevance` |
| `submission/corpus.csv` | 99,567 | `passage_id, title, text, source_url` |

`corpus.csv` is **not in this repository** (about 169 MB). See [Reproducing the results](#reproducing-the-results).

- **Corpus:** Bengali Wikipedia (`sagorsarker/bangla-wikipedia`, snapshot 20250620), split on sentence boundaries into passages of mostly 80 to 300 words.
- **Queries:** written by language models from one passage each, then filtered by an automatic answerability check.
- **Relevance:** one gold passage per query.
- **Splits:** `train` 7,246, `dev` 1,564, `test` 1,454, assigned by passage so that no passage appears in two splits.
- **Script labels:** based on characters, not language. An English word written in Bangla script counts as `bangla`.

The first letter of `query_id` gives the part:

| Part | Queries | What it holds |
|---|---|---|
| `P` (parallel) | 3,441 (1,147 passages) | The same question in all three forms. IDs end in `_b`, `_m`, `_r`. |
| `G` (non-parallel) | 6,823 | The three forms of a passage usually ask different things. |

Use the `P` part for any claim about the effect of the writing form.

The data card is in [`submission/README.md`](submission/README.md). Its baseline numbers come from an earlier run; the table below replaces them.

## Results

nDCG@10 on the parallel test split: 165 questions in three forms, 495 queries.

| System | Bangla | Mixed | Romanized | All |
|---|---|---|---|---|
| BM25 | 0.575 | 0.348 | 0.014 | 0.312 |
| Dense (`multilingual-e5-base`) | 0.708 | 0.659 | 0.137 | 0.501 |
| Hybrid, dense weight 0.7 | 0.736 | 0.622 | 0.079 | 0.479 |
| Hybrid 0.7 + normaliser | 0.737 | 0.651 | **0.248** | 0.545 |
| Single tuned weight (0.8) + normaliser | **0.752** | **0.690** | 0.238 | **0.560** |
| Per-script weights + normaliser | 0.752 | 0.690 | 0.233 | 0.558 |

Intervals below are 95% paired bootstrap intervals.

- **Writing form matters.** On the same questions, quality falls from Bangla script to mixed to Romanized. BM25 almost fails on Romanized queries.
- **Dense retrieval narrows the mixed gap, not the Romanized one.**
- **The normaliser helps Romanized queries.** It raises the 0.7 hybrid from 0.079 to 0.248 (+0.169, [+0.119, +0.221]).
- **Per-script fusion weights do not beat one tuned weight.** The overall difference is -0.002, [-0.017, +0.014].

Full per-query ranks and tables are in [`corpus_md_build/data/release_eval/`](corpus_md_build/data/release_eval/).

## Method

1. **BM25** (`bm25s`) on lower-cased tokens, without stemming.
2. **Dense retrieval** with `intfloat/multilingual-e5-base`.
3. **Skeleton normaliser** for Latin tokens: each corpus Bangla word is reduced to a consonant skeleton, and a Latin query token is expanded with the 3 most frequent words sharing its skeleton. The original token is kept. It needs no training data or relevance labels.
4. **Fusion:** reciprocal rank fusion, or a weighted sum of min-max normalised scores. Tuned weights are chosen on training queries only.

## Setup

Python 3.12. Run from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install bm25s==0.3.16 tqdm        # used by corpus_md_build/, not yet in requirements.txt
cp .env.example .env                  # API keys; never commit .env
python scripts/check_setup.py
```

API keys are needed only to generate new queries (`ANTHROPIC_API_KEY`, `GROQ_API_KEY`), which also needs `pip install anthropic`. Evaluation needs no key. Scripts use Apple `mps` when available and otherwise the CPU.

## Reproducing the results

The corpus and the passage embeddings are too large for the repository, so they have to be present locally:

| File | Where it comes from |
|---|---|
| `corpus_md_build/data/corpus.jsonl` | The working corpus (100,000 passages) |
| `submission/corpus.csv` | `26_build_submission.py` |
| `corpus_md_build/data/emb_e5base_passages.npy` | `07_encode_e5.py` |

**A fresh clone cannot yet rebuild the exact corpus.** The working corpus holds 433 passages from two government websites, which are excluded from the release because those sites state no open licence. They sit in the middle of the passage numbering, so a rebuild without them gives different passage IDs that no longer match `qrels.csv`. Ask the repository owner for the corpus files.

With `corpus.jsonl` in place:

```bash
cd corpus_md_build
python 26_build_submission.py                          # rewrites the three files in submission/
python 07_encode_e5.py                                 # passage embeddings
python 29_eval_release.py .. data/release_eval/release_eval.json
python 30_cluster_bootstrap.py data/release_eval ..
python 31_derived_tables.py data/release_eval ..
python 32_check_release.py ..                          # recompute dataset statistics
```

These commands overwrite the tracked files in `submission/` and `data/release_eval/`.

## Repository layout

```
corpus_md_build/     benchmark pipeline (numbered scripts) and its data
submission/          released queries, labels and data card
review/              human review sheets and flagged queries
scripts/             earlier pilot-stage scripts and setup check
data/                pilot notes, coverage files and corpus card
experiments/         pilot results and experiment tracker
dataset/             earlier template-generated query set
notebooks/           query filtering notebook
paper/               LaTeX drafts
research_proposal.md original research plan
corpus.md            early corpus build notes
```

Main scripts in `corpus_md_build/`:

| Scripts | Purpose |
|---|---|
| `01`–`03` | Download Wikipedia, crawl web pages, build the corpus |
| `12`, `13` | Generate and filter the non-parallel queries |
| `21`, `21b`, `22`, `25` | Generate and clean the parallel queries |
| `15`, `17`, `19`, `20`, `27`, `28` | Human review sheets, corrections and flags |
| `26` | Build the release files |
| `29`–`32` | Evaluate and check the release |

Other numbered scripts are earlier iterations or side analyses.

## Limitations

- Queries are model-written and mostly **not reviewed by people**. They are likely cleaner than real user questions.
- In a reviewed sample of 34 parallel passages (one reviewer), 9 had a question that does not name what it asks about, so several passages could answer it.
- With one gold passage per query, other correct passages count as wrong, so scores are lower bounds.
- The corpus is encyclopedic and small, with little text on fees, procedures or local services.
- Only one dense model was tested, and the test split had been scored in earlier analyses.

## Licence

- Passage text is from Wikipedia under **CC BY-SA 4.0**. Keep `title` and `source_url` for attribution.
- A licence for the queries, labels and code **has not been chosen yet**.
