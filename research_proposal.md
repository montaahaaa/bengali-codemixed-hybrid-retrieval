# Research Proposal

## A Benchmark and Hybrid Retrieval Framework for Bengali Code-Mixed Queries in Retrieval-Augmented Generation

Status: proposal. No benchmark, dataset, or experimental result exists yet. Everything below describes planned work.

---

## 1. Summary

Bengali speakers routinely mix Bengali and English in one sentence, and many write Bengali in Latin script (Romanized Bengali), for example "passport renew korte koto taka lagbe?". Retrieval-augmented generation (RAG) systems depend on a retriever that finds the right passages for a query. Retrievers are built and tested almost entirely on monolingual, single-script queries. How badly they handle code-mixed Bengali is unknown, because no retrieval benchmark for this setting exists.

This project has two parts. First, build a benchmark of code-mixed Bengali-English queries with graded relevance judgments over a passage collection. Second, design and test a hybrid (sparse plus dense) retrieval framework that adapts to how mixed each query is. The benchmark is the prerequisite: without it, neither the problem nor any fix can be measured.

## 2. Background and motivation

- Bengali has roughly 280 million speakers but is low-resource in NLP.
- In digital writing, code-mixing and Romanization are the norm, not the exception. Bengali typing needs a special keyboard layout, so many users type Latin letters.
- RAG grounds a language model in retrieved text. If retrieval fails, generation fails, and for non-English languages retrieval is often the main bottleneck.
- Two failure modes are expected for code-mixed queries:
  1. **Vocabulary split.** Query terms come from two languages, so a lexical matcher like BM25 can match only part of the query in any single-language passage.
  2. **Script mismatch.** A Romanized Bengali word shares no characters with its Bengali-script form, so lexical overlap with a Bengali-script passage is zero even when meaning matches perfectly.
- Dense multilingual encoders may handle this better in principle, but they are trained mostly on clean monolingual text, and their behavior on Romanized, mixed input is unmeasured for Bengali.

## 3. Related work (short)

- **Code-mixing benchmarks** such as GLUECoS and LinCE cover classification-style tasks (language ID, NER, sentiment, QA, NLI) for pairs like Hindi-English and Spanish-English. Bengali-English and retrieval are not covered.
- **Multilingual retrieval benchmarks** such as MIRACL and mMARCO include Bengali, but queries are monolingual.
- **Retrieval methods**: BM25; dense retrieval (DPR, ColBERT, E5, BGE-M3); hybrid fusion (Reciprocal Rank Fusion, convex combination of normalised scores); cross-encoder reranking.
- **Code-mixing metrics**: Code-Mixing Index (CMI) and switch-point measures describe how much and how finely two languages are interleaved.
- **Multilingual RAG** work shows retrieval quality is the main bottleneck for non-English input, and that retrievers can be biased toward English. It assumes the query has one identifiable language.

## 4. Research gaps

1. **No benchmark.** There is no retrieval benchmark with code-mixed Bengali-English queries and relevance judgments.
2. **No tailored retriever.** Hybrid fusion methods apply the same weights to every query, ignoring how mixed or how Romanized the query is.
3. **No RAG evaluation on code-mixed input.** It is unknown how retrieval degradation affects final answer quality.

## 5. Research questions and hypotheses

| ID | Research question | Working hypothesis |
|----|-------------------|--------------------|
| RQ1 | How much do BM25, dense, and hybrid retrievers degrade on code-mixed queries compared with monolingual Bengali and monolingual English versions of the same information need? | Both degrade. BM25 degrades more, especially on Romanized queries. |
| RQ2 | Does retrieval quality fall as code-mixing intensity rises, and is the relationship monotonic? | Quality falls with intensity, but script (Romanized or not) explains more than CMI alone. |
| RQ3 | How much does query normalisation (transliteration, expansion) recover? | It helps the sparse branch noticeably and the dense branch very little. |
| RQ4 | Does adapting fusion weights to query properties beat fixed fusion (RRF, fixed weights)? | A small but consistent gain, concentrated in high-mixing and Romanized queries. |
| RQ5 | Do retrieval gains carry through to RAG answer quality? | Partly. Gains shrink after generation because strong generators can compensate for some retrieval noise. |

These are hypotheses to be tested, and each may turn out false. A null or negative result on RQ3 or RQ4 is still a valid finding.

## 6. Objectives

1. Design and build a code-mixed Bengali-English retrieval benchmark with graded relevance and per-query code-mixing metadata.
2. Measure the code-mixing penalty on standard sparse, dense, and hybrid retrievers.
3. Design a hybrid retrieval framework with script-aware query normalisation, query-adaptive fusion, and reranking.
4. Evaluate the framework alone and inside a RAG pipeline, with ablations and significance tests.
5. Release the benchmark, code, and configurations.

## 7. Methodology

### 7.1 Overall design

Two artefacts, built in order: benchmark first, framework second. One rule governs the whole project: **freeze the corpus, queries, and all systems before relevance annotation starts.** Pooled judgments only cover what the pooled systems retrieved, so changing a system afterward biases its scores down.

### 7.2 Benchmark construction

**Design principles**

- Queries should reflect real information needs expressed the way users actually write.
- The collection should cover the full range of mixing intensity, from light (one English word) to heavy (alternating tokens).
- Cover several domains so results are not tied to one vocabulary.
- Use graded relevance (0, 1, 2), not binary.
- Include controlled variants of the same information need so that RQ1 is a fair comparison.

**Query sourcing (options to decide in the pilot)**

| Option | Pros | Risks |
|--------|------|-------|
| Mine public social-media text (YouTube, Reddit, Facebook groups, Q&A forums) | Naturalistic | Most comments are not information-seeking, so yield is low and many queries have no answer in the corpus. Platform terms and privacy need care. |
| Ask native speakers to write the question they would type into a chatbot, given a topic or a passage | Guaranteed answerable, controllable coverage | Elicited mixing is cleaner than spontaneous mixing |
| Hybrid: mine for style and topics, then have annotators rewrite into answerable queries | Balances both | More annotation cost |

The pilot will compare yield and realism of these options on a small sample before committing. The recommended default is the hybrid option, with provenance recorded per query (mined, elicited, rewritten).

**Controlled variants.** For a subset of queries, annotators produce parallel monolingual Bengali (Bengali script), monolingual English, and Romanized-only versions of the same information need. This lets RQ1 and RQ2 isolate the effect of mixing from the effect of topic difficulty.

**Language identification.** Token-level language tags are needed for all code-mixing metrics. Script alone is not enough: Romanized Bengali is entirely Latin script, so a script-based tagger labels it English and the heavily mixed queries disappear from the data. Tagging must be done at language level. Plan: start with a lexicon-based tagger, then compare against an existing or fine-tuned token-level language identifier, and validate on a manually tagged sample. Report tagging accuracy.

**Metadata per query**: token-level language tags, CMI, switch-point fraction, script profile (Bengali, Latin, mixed), domain, provenance.

**Passage collection.** Needs to be bilingual, since code-mixed queries contain English content terms, and must actually contain answers to the benchmark queries. Wikipedia (Bengali and English) is a convenient base but may not cover everyday questions such as fees, procedures, and local services. Plan: use Wikipedia plus additional openly licensed Bengali and English web text, and check answer coverage in the pilot. If coverage is poor, build the corpus around the queries (add documents targeted at the query topics, plus many distractors).

**Relevance annotation**

- Depth-k pooling over all systems in the study (k to be fixed in the pilot, for example 10 to 20).
- Three-level grades. Grade by content only, never by passage language.
- Two independent annotators per pair, a written guideline, a calibration round, and a qualification round.
- Disagreement rule: equal grades accepted; adjacent grades resolved by a stated rule; large splits sent to a third judge.
- Agreement measured with ordinal Krippendorff's alpha.
- Budget check: the number of judgments is roughly queries x systems x depth, minus overlap. This must fit annotator capacity. LLM-assisted pre-judging with human verification can reduce cost, but must itself be validated against human labels on a sample.

**Benchmark size.** Target is to be set from the pilot and annotation budget. A reasonable aim is a few hundred queries, enough for stratified analysis (at least about 50 per stratum) and for significance testing. The target is a plan, not a result.

### 7.3 Baselines

- BM25 on raw query.
- Dense retrieval with multilingual encoders (for example multilingual E5 and BGE-M3), raw query.
- Hybrid with Reciprocal Rank Fusion.
- Hybrid with fixed convex weights (for example 0.3, 0.5, 0.7).
- Additional strong baselines: LLM-based query rewriting or transliteration into Bengali script and into English before retrieval; BGE-M3 using its own sparse plus dense modes.

All systems share one index and one implementation of fusion primitives so differences come from the method, not from incidental tooling.

### 7.4 Proposed framework

Applied in order to each query:

1. **Language and script profiling.** Token tags, CMI, switch-point fraction, script profile.
2. **Script-aware normalisation (sparse branch).** Additive expansion: keep the original token and add its Bengali-script (or English) counterpart. Options for the mapping: a curated table, a transliteration model, or an LLM. Compare them.
3. **Multi-branch retrieval.** BM25 on the normalised query, dense encoder on the raw query.
4. **Query-adaptive fusion.** Min-max normalise each branch, then combine with a weight that depends on the query profile. Two variants:
   - A hand-set schedule (the hypothesis in its simplest form).
   - A learned weight (for example a small regression or classifier on profile features), trained with cross-validation on the judged queries.
   The comparison against fixed weights and RRF is the real test of RQ4.
5. **Cross-encoder reranking** of the top candidates, with a multilingual reranker. Off-the-shelf English rerankers may transfer poorly to Bengali, so a multilingual one should be a baseline, and fine-tuning on code-mixed pairs is an option if data allows.

### 7.5 Evaluation

- **Metrics:** Recall@{5, 10, 20, 100}, MRR@10, MAP, NDCG@10 (primary, because it uses graded relevance).
- **Pooling bias control:** report the share of each run's top 10 that carries a judgment, next to every result.
- **Stratified analysis:** by CMI band, script profile, domain, and query provenance.
- **Ablations:** remove each component in turn (normaliser, adaptive fusion, reranker).
- **Significance:** paired bootstrap on per-query NDCG@10 differences, with effect sizes and confidence intervals. With a few hundred queries, small gains may not be detectable, and the write-up must say so.
- **Overfitting guard:** any weight or threshold tuned on judged queries must use held-out folds. Report test results only on data not used for tuning.
- **Separation of runs from scoring:** retrieval produces ranked run files; one evaluation script computes every metric from the judgments.

### 7.6 End-to-end RAG evaluation

- Fix a generator and prompt. Vary only the retriever.
- Evaluate answer quality on a sample of queries: correctness against reference answers or key facts, faithfulness to retrieved passages, and answer language appropriateness.
- Use human judgments on a subset to validate any LLM-as-judge scoring.
- Report which failures come from retrieval and which from generation.

## 8. Critical assessment of the idea

**Strengths**

- A real, specific gap. The script-mismatch problem for Romanized Bengali is concrete and testable.
- The benchmark is useful even if the proposed framework does not win. Negative or small results on the framework are still publishable alongside a solid benchmark.
- The design supports clear, falsifiable questions.

**Weaknesses and risks, with mitigations**

| Risk | Why it matters | Mitigation |
|------|----------------|------------|
| Mined social-media comments are rarely real information needs | Many candidate queries have no answer in any corpus, producing empty judgments | Pilot yield first; use elicited or rewritten queries; check answerability before annotation |
| Corpus does not contain answers to everyday questions | Retrieval metrics become meaningless | Build or extend the corpus around the query topics; measure answer coverage in the pilot |
| Annotation cost and quality | Dominant cost and the main source of validity | Small pool depth, clear guidelines, calibration, agreement reporting, optional validated LLM pre-judging |
| Lexicon-based language ID is crude | Errors propagate to CMI and strata | Validate on a manual sample; compare with a learned tagger |
| Novelty of the framework is modest | Adaptive fusion and transliteration expansion are incremental | Frame the benchmark as the main contribution; frame the framework as a tested hypothesis with honest ablations |
| Hand-set fusion weights look arbitrary | Reviewers will ask where the numbers came from | Include a learned-weight variant with cross-validation |
| Small query set limits significance | Small gains not detectable | Report effect sizes and intervals; do not claim gains the data cannot support |
| Stronger LLM-based baselines may beat the framework | LLM query rewriting is an obvious competitor | Include it as a baseline from the start |
| Scope creep into RAG generation | Time and cost | Keep RAG evaluation to a fixed generator and a sample |

**Overall judgment.** Feasible and worthwhile if scoped as: a carefully built benchmark (primary contribution), a clear measurement of the code-mixing penalty, and a modest, honestly evaluated framework. The weakest part of the plan as typically conceived is query and corpus design, not the retrieval model, so the pilot study deserves the most attention.

## 9. Expected contributions

1. The first retrieval benchmark for Bengali-English code-mixed queries with graded relevance and code-mixing metadata (planned).
2. An empirical measurement of how sparse, dense, and hybrid retrievers degrade with code-mixing and Romanization.
3. A documented methodological point: script-based language identification removes Romanized Bengali and therefore the most mixed queries from code-mixed corpora. The size of this effect will be measured on our own data.
4. A hybrid retrieval framework with ablations and an honest comparison with strong baselines.
5. An end-to-end RAG evaluation on code-mixed Bengali input.
6. Released data, code, and configurations.

## 10. Work plan (indicative, about 10 months)

| Phase | Months | Work | Output |
|-------|--------|------|--------|
| 1. Pilot | 1 to 2 | Compare query sourcing options; check corpus answer coverage; draft guidelines; test language ID | Decisions on sourcing, corpus, pool depth |
| 2. Benchmark build | 3 to 5 | Collect queries and variants; build corpus; freeze; first pass of language ID validation | Frozen queries and corpus |
| 3. Systems and pooling | 5 to 6 | Implement baselines and framework; generate runs; build pools | Run files, pools |
| 4. Annotation | 6 to 8 | Train annotators; judge; adjudicate; measure agreement | Relevance judgments |
| 5. Evaluation | 8 to 9 | Baselines, ablations, stratified analysis, significance tests, tuning with held-out folds | Results tables |
| 6. RAG evaluation and writing | 9 to 10 | Fixed-generator study; write-up; release | Thesis, paper, data release |

Annotation timing is the main schedule risk. Systems must be frozen before phase 4.

## 11. Resources

- Compute: a single GPU for encoding the corpus and running rerankers is sufficient; encoder embeddings are cached.
- Software: `rank_bm25` or Pyserini, `sentence-transformers`, FAISS, a TREC-format evaluation tool.
- People: native Bengali annotators with good English, at least two per judgment.
- Possible API costs for LLM-based baselines and the RAG generator.

## 12. Ethics and licensing

- Mined social-media text must follow platform terms. Remove usernames and identifying details, and release only what licensing allows (possibly only identifiers and derived annotations).
- Annotators are paid fairly and informed about content.
- Wikipedia and other corpus sources are used under their licenses; state them in the release.
- Document known biases: online, urban, younger-writer skew in code-mixed text; domain coverage limits.

## 13. Limitations (stated in advance)

- Bengali-English only; other mixed pairs are out of scope.
- Text only; speech and ASR are out of scope.
- The benchmark will be small compared with large English retrieval benchmarks.
- Romanization has no standard spelling, so no tagger or normaliser will be complete.
- Results depend on the chosen corpus and pooled systems.

## 14. Selected references

- Aguilar et al. (2020). LinCE: A Centralized Benchmark for Linguistic Code-switching Evaluation.
- Bhattacharjee et al. (2022). BanglaBERT: Language Model Pretraining and Benchmarks for Low-Resource Language Understanding Evaluation in Bangla.
- Bonifacio et al. (2021). mMARCO: A Multilingual Version of the MS MARCO Passage Ranking Dataset.
- Chen et al. (2024). BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings.
- Cormack, Clarke, Buettcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods.
- Das and Gamback (2014). Identifying Languages at the Word Level in Code-Mixed Indian Social Media Text.
- Guzman et al. (2017). Metrics for Modeling Code-Switching Across Corpora.
- Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering.
- Khanuja et al. (2020). GLUECoS: An Evaluation Benchmark for Code-Switched NLP.
- Krippendorff (2011). Computing Krippendorff's Alpha-Reliability.
- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
- Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
- Thakur et al. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.
- Voorhees (2000). Variations in relevance judgments and the measurement of retrieval effectiveness.
- Wang et al. (2022). Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5).
- Zhang et al. (2023). MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages.

Verify bibliographic details before citing in a submission.