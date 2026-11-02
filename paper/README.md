# Paper

Target: IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP). LaTeX, `IEEEtran` journal class. American English.

```
main.tex            preamble, title, abstract, \input of every section
sections/           one .tex per section (01 intro ... 09 conclusion)
references.bib      BibTeX; entries marked VERIFY are unchecked
figures/ tables/    generated figures and tables
```

## Rules

- A section is written only when its part of the project is finished (roadmap stage done) or when asked.
- Every number comes from a computed run. Anything not yet known uses `\pending{...}` (prints red). Before submission, `grep -r "pending" .` must return nothing.
- Verify every reference, then remove its `VERIFY` note.
- Write the abstract and conclusion last.

## Build

Needs a LaTeX install (not present on this machine yet) or Overleaf. Run: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.

Check the journal's current author guidelines for page limits and the template before submission.
