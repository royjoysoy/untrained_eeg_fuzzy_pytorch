# Short paper draft

Working title: **How stable are untrained CNNs? Measuring random-weight CNN variability on EEG with Fuzzy PyTorch**

Draft in `draft.md` (Markdown so anyone can edit it in a PR). Figures come from
`scripts/04_compare.py` (`results/figures/`). Copy final figures here only at submission time.

## Outline

1. **Introduction** — untrained/random-weight CNNs as EEG baselines and feature extractors;
   two sources of variability (initialization, floating-point); why it matters for reproducibility.
2. **Methods**
   - Data: ds003478 v1.1.0, run-01, 1-min eyes-open/closed blocks, 2 s windows, 60 channels,
     1–40 Hz, 128 Hz, average reference; 113 participants (74 low / 39 high BDI), 544 excluded.
   - Models: small 1D CNN, ShallowConvNet, EEGNet (braindecode), random weights, eval mode.
   - Probe: logistic regression, stratified 5-fold CV with a fixed split.
   - Seed variability: N seeds. Numerical variability: Fuzzy PyTorch, M MCA samples at a fixed seed.
   - Metrics: AUC spread; per-participant SD and significant digits of P(high BDI).
3. **Results** — Figure 1: AUC per run; Figure 2: significant digits; Table 1: `results/summary.csv`.
4. **Discussion** — how seed variability compares with numerical variability; implications for
   reporting untrained-CNN baselines; limitations (one dataset, small N, probe choice).
5. **Contributors and acknowledgements** — Brainhack Montreal 2026; Alliance / Rorqual compute.
