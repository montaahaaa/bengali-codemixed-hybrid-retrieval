# Reading check (roadmap 1.3)

20 passages drawn with seed 42 from `corpus.jsonl` (10 Bengali, 10 English by chance of the draw) and read by eye.

## Findings
- No sentence is cut in the middle. Text is well formed in both languages; no leftover wiki markup or tables seen.
- Section headings are joined inline with the following text (for example "জীবনী মরিয়ম টিফ্লিস এ জন্মগ্রহণ করেছিলেন"). Harmless for retrieval; noted, not fixed.
- Passages from the end of an article can be short (one was 83 words). 24.0% of Bengali and 15.9% of English passages are under 100 words, and a few are 1 word long. These are counted in `stats.json`; they were not merged, to keep the "never exceed 150 words" rule.
- A few Bengali articles contain mixed digit scripts (Latin 1682 beside Bengali ১৯৭১). Left as is.
- Most passages are general encyclopedic text (history, places, biographies, railways, software). Few look like they answer everyday procedure or fee questions. Whether the corpus answers such questions is exactly what phase 1.5 measures. This is an observation from 20 passages, not a finding.

## Not done
- Only 20 passages were read. This does not prove the whole corpus is clean.
