# Paper analyses, group A: a robust main comparison (#30)

Seeds vs stochastic rounding on **all windows**, with more runs, a crossed design and
data-split variability. Nothing here changes the pilot outputs: everything is written under
`results/paper/` (config `configs/paper.yaml`, 30 seeds and 30 rounding samples).

Pipeline: `extract_gpu_features.py` runs the CNNs on the GPU (Turbulence op mode, every
float add/sub/mul/div/fma stochastically rounded) and saves features averaged per
participant and 1-minute block, one file per run; `probe_runs.py` probes every saved run
(and any number of CV splits) and writes tidy CSV; `variability_stats.py` computes the
per-participant statistics. Features are saved once and reused by groups B and C.

Environment: GB10 (aarch64, sm_121) with an op-mode Turbulence install
(`TURBULENCE_IREE_BUILD`, `TURBULENCE_CLANG`); the EEG cache from `scripts/01_*`.
GPU cost: ~2–4 s per run of 19 500 windows, plus ~3 s per compilation.

## A1. All windows, 30 seeds and 30 rounding samples

```bash
python scripts/extract_gpu_features.py --config configs/paper.yaml --weight-seeds 0-29 --rn
python scripts/extract_gpu_features.py --config configs/paper.yaml --weight-seeds 0 --sr-seeds 0-29
python scripts/probe_runs.py --config configs/paper.yaml --features results/paper/gpu_features \
    --out results/paper/A1 --only $(printf 'w%03d_rn ' $(seq 0 29)) $(printf 'w000_sr%03d ' $(seq 0 29))
```

Fuzzy PyTorch on CPU (110–190 ms per window) is the cross-check from the pilot; at all
windows 30 samples cost ~70 CPU-hours, so it is left to a SLURM array
(`03_mca_variability.py --sample i`, one task per sample) if needed.

## A2. Metrics

`variability_stats.py` reports, per participant × model × condition × source, the SD of
P(high BDI) across runs and its significant digits with two estimators:
`sig_digits_parker` = −log10(σ/|μ|) (Parker 1997; 15.95 when all runs agree), and
`sig_digits_cnh` (Sohier et al. 2021, 95 % probability and confidence) from Verificarlo's
`significantdigits` package when installed (`pip install significantdigits`).

## A3. Crossed design: weight seeds × rounding seeds

```bash
python scripts/extract_gpu_features.py --config configs/paper.yaml --weight-seeds 0-9 --sr-seeds 0-9
python scripts/probe_runs.py --config configs/paper.yaml --features results/paper/gpu_features \
    --out results/paper/A3 --only $(for w in $(seq 0 9); do for k in $(seq 0 9); do printf 'w%03d_sr%03d ' $w $k; done; done)
```

Per participant, a two-way crossed ANOVA (one run per cell, method of moments) splits the
variance of P(high BDI) into weight seed, rounding seed and residual (seed × rounding +
noise). The rounding seed has no shared meaning across weight seeds, so numerical
variability is the rounding + residual part. `rounding_sd_min/median/max` show whether the
rounding sensitivity depends on the weight draw.

## A4. Data-split variability, and a perturbed probe

```bash
python scripts/probe_runs.py --config configs/paper.yaml --features results/paper/gpu_features \
    --out results/paper/A4 --only w000_rn --cv-seeds 0-29
python scripts/variability_stats.py --a1 results/paper/A1 --a3 results/paper/A3 --a4 results/paper/A4 \
    --out results/paper/A_stats
```

The probe itself (scikit-learn) is never instrumented, so `probe_under_rounding.py`
re-implements it in PyTorch (same standardization, same objective as sklearn's default
`LogisticRegression`, L-BFGS in float64, same split) and runs it on the Fuzzy CPU build:
features from round-to-nearest (`w000_rn`, probe-only perturbation) or from a GPU rounding
run (`w000_srKKK`, end to end). A few seconds per sample:

```bash
for k in $(seq 0 29); do
  MODE=sr SEED=$((2001 + k)) bash environment/fuzzy_arm.sh scripts/probe_under_rounding.py \
    --config configs/paper.yaml --features results/paper/gpu_features --features-run w000_rn \
    --sample $k --out results/paper/A4_probe_sr/probe_only
  MODE=sr SEED=$((3001 + k)) bash environment/fuzzy_arm.sh scripts/probe_under_rounding.py \
    --config configs/paper.yaml --features results/paper/gpu_features --features-run $(printf 'w000_sr%03d' $k) \
    --sample $k --out results/paper/A4_probe_sr/end_to_end
done
```

Note: with its default tolerance (`tol=1e-4`) sklearn's probe stops up to 1.8e-3 away from
the optimum in P(high BDI) (cnn1d, eyes closed); the PyTorch probe agrees with a tightly
converged sklearn (`tol=1e-12`) to 4e-7. The pilot and `probe_runs.py` keep the default.

```bash
python scripts/summarize_probe_rounding.py --dir results/paper/A4_probe_sr --out results/paper/A_stats
```

## Results (2026-10-07, all windows, 113 participants)

Median over participants of the SD of P(high BDI) across runs, and share of participants
whose predicted class changes (flip). 30 runs per source; seeds and data splits use
round-to-nearest; "rounding" = GPU stochastic rounding of the CNN, weight seed 0, sklearn probe.

| model, condition | seeds: SD / flip | data splits: SD / flip | rounding: SD / flip | rounding, sig. digits (CNH) |
|---|---|---|---|---|
| cnn1d closed | 0.157 / 74 % | 0.092 / 52 % | 1.8e-7 / 0 % | 5.7 |
| cnn1d open | 0.034 / 30 % | 0.070 / 44 % | 4.1e-8 / 0 % | 6.6 |
| shallow closed | 0.215 / 84 % | 0.131 / 63 % | 1.1e-4 / 0 % | 2.8 |
| shallow open | 0.207 / 91 % | 0.132 / 67 % | 1.2e-4 / 0 % | 3.0 |
| eegnet closed | 0.247 / 89 % | 0.143 / 60 % | 7.1e-7 / 0 % | 5.0 |
| eegnet open | 0.111 / 77 % | 0.111 / 56 % | 6.8e-8 / 0 % | 6.2 |

- Seed variability exceeds rounding variability by 3–6 orders of magnitude; no participant's
  prediction ever flips under rounding (30 samples, all models and conditions).
- **The data split matters about as much as the seed** (SD 0.07–0.14, 44–67 % flips): the
  CV split is a variability source of the same size and should be reported.
- **Crossed design (A3, 10 × 10):** the weight seed explains essentially all the variance
  (median share ≈ 1.0). The rounding sensitivity itself depends strongly on the weight
  draw: a participant's rounding SD varies by a median factor of 570 (cnn1d), 1 200–1 400
  (shallow) and 3 300 (eegnet) across the 10 weight seeds, eyes closed. With eyes open some
  weight seeds give exactly zero rounding SD for some participants, so the ratio is undefined.
- **Probe under rounding (A4):** perturbing only the probe moves P(high BDI) by 1e-9–2e-8;
  end to end (rounded features + rounded, tightly converged PyTorch probe) by 3e-8–3.4e-7,
  0 flips. With sklearn's default probe on the same rounded features, ShallowFBCSPNet
  moves by 1.1e-4, ~300× more (eegnet ~10×, cnn1d ~2×): most of the "rounding" effect seen
  for ShallowFBCSPNet comes from the loose stopping tolerance of the probe amplifying tiny
  feature differences.

Tables: `results/paper/A_stats/` (`summary.csv`, `variance_summary.csv`,
`summary_probe_sr.csv`, per-participant CSVs).
