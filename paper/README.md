# Paper

LaTeX source of the paper: `main.tex` and `references.bib`. Red `[TODO: ...]` marks what is
still open. Build it in this folder:

    latexmk -pdf main.tex

The figures in `figures/` come from the saved results, without running any model:

    python scripts/paper_figures.py     # from the repository root, after phase 1, phase 2 and permutation.py

Every number in the text and tables comes from `results/` (phase 1 and 2 metrics,
`results/stats/`); a comment above each table names its source. Edit the text by pull request
so changes stay reviewable.
