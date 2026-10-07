# Paper analyses, group C: fixing the interpretation problem (#32)

The BDI probe is close to chance, so seed and rounding variability are measured on a task
the features may not solve. Group C checks that the features carry EEG information, how far
BDI performance is from chance, how it compares with a classical baseline, and whether sex
or anxiety explain it. All scripts reuse the features saved by group A
(`results/paper/gpu_features`, see `docs/paper_A.md`) and write to `results/paper/C*`.
CPU only, a few minutes each.

## C8. Positive control: eyes closed vs eyes open

```bash
python scripts/positive_control.py --config configs/paper.yaml --features results/paper/gpu_features \
    --out results/paper/C8 --only $(printf 'w%03d_rn ' $(seq 0 29)) $(printf 'w000_sr%03d ' $(seq 0 29))
python scripts/variability_stats.py --a1 results/paper/C8 --out results/paper/C8_stats
```

Each participant-block (1 minute) is classified eyes closed vs open; CV is stratified by
condition and grouped by participant (`StratifiedGroupKFold`). Output rows use
`participant_id = <participant>_b<block>`, so `variability_stats.py` gives the same
seed vs rounding statistics on this task.

## C9. Label-permutation null

```bash
python scripts/permutation_null.py --config configs/paper.yaml --features results/paper/gpu_features \
    --out results/paper/C9 --n-perm 1000 --workers 16
```

Weight seed 0, round to nearest, fixed CV split (as in the pilot): observed AUC vs 1000
permutations of the BDI labels through the same probe; p = (1 + #null ≥ observed) / 1001.

## C10. Band-power baseline

```bash
python scripts/bandpower_baseline.py --config configs/paper.yaml --out results/paper/C10 --n-perm 1000
```

Log10 band power (delta 1–4, theta 4–8, alpha 8–13, beta 13–30 Hz) per channel from the
same 2 s windows (Welch, one 2 s segment, spectra averaged per participant and condition):
240 features, same probe, split and permutation test.

## C11. Confounds: sex and trait anxiety

```bash
python scripts/confounds.py --config configs/paper.yaml --features results/paper/gpu_features \
    --weight-seeds 0-29 --bandpower results/paper/C10/bandpower_features.npz --out results/paper/C11
```

Over the 30 round-to-nearest seeds: BDI with sex regressed out of the features inside each
fold (fitted on training participants only), BDI within women and within men, STAI median
split as the target, and sex alone as the only feature.

## C12. Candidate second datasets (not downloaded)

Checked on OpenNeuro (2026-10-07, metadata only). All CC0, BIDS EEG with a resting task.

| Dataset | Label | Subjects | Why | To check before use |
|---|---|---|---|---|
| [ds003490](https://openneuro.org/datasets/ds003490) | Parkinson's vs control | 50 | Same lab (Cavanagh) and likely the same 64-channel system and eyes-open/closed rest protocol as ds003478: the pipeline should transfer almost unchanged | trigger codes, group sizes, ON/OFF medication sessions |
| [ds004584](https://openneuro.org/datasets/ds004584) | Parkinson's vs control | 149 | Largest sample; rest, eyes open | group sizes, channel count, recording length |
| [ds003944](https://openneuro.org/datasets/ds003944) | First-episode psychosis vs control | 82 | Psychiatric label, like BDI | montage, eyes open/closed markers |
| [ds004504](https://openneuro.org/datasets/ds004504) | Alzheimer's / FTD / control | 88 | Strong clinical effect expected; eyes closed | 19-channel 10-20 montage: models need `n_chans=19` |
| [ds002778](https://openneuro.org/datasets/ds002778) | Parkinson's vs control | 31 | Small, quick sanity check | 32 channels, ON/OFF sessions |

Recommendation: ds003490 first (same acquisition as ds003478), then ds004584 for sample size.

## Results (2026-10-07, all windows, 113 participants)

**C8, positive control.** Eyes closed vs open is easy for the random-CNN features, so they do
carry EEG information, and the seed vs rounding gap holds on a task that works:

| model | AUC, 30 seeds | AUC, 30 rounding samples | block P(closed): SD / flips, seeds | rounding |
|---|---|---|---|---|
| cnn1d | 0.929 ± 0.013 | 0.934 (identical) | 0.078 / 37 % | 5.2e-8 / 0 % |
| shallow | 0.947 ± 0.014 | 0.942 ± 0.0001 | 0.115 / 49 % | 1.7e-4 / 0.15 % |
| eegnet | 0.885 ± 0.023 | 0.907 (identical) | 0.135 / 64 % | 5.7e-8 / 0 % |

**C9, permutation null** (weight seed 0, 1000 permutations):

| | cnn1d | shallow | eegnet |
|---|---|---|---|
| eyes closed | 0.654, **p = 0.022** | 0.582, p = 0.15 | 0.541, p = 0.30 |
| eyes open | 0.464, p = 0.67 | 0.523, p = 0.37 | 0.599, p = 0.08 |

Null 95th percentile ≈ 0.61–0.63. Only cnn1d eyes closed beats chance, and not after correcting
for the 6 tests (Bonferroni 0.05 / 6 = 0.008).

**C10, band power:** AUC 0.529 (closed, p = 0.36), 0.592 (open, p = 0.12): no better than the
random CNNs.

**C11, confounds** (mean AUC over 30 seeds):

| model, condition | BDI | sex regressed out | women | men | STAI median split | sex alone |
|---|---|---|---|---|---|---|
| cnn1d closed | 0.636 | 0.629 | 0.515 | 0.776 | 0.620 | 0.537 |
| cnn1d open | 0.450 | 0.457 | 0.583 | 0.568 | 0.469 | 0.537 |
| shallow closed | 0.553 | 0.542 | 0.465 | 0.757 | 0.590 | 0.537 |
| shallow open | 0.580 | 0.583 | 0.614 | 0.446 | 0.569 | 0.537 |
| eegnet closed | 0.533 | 0.562 | 0.505 | 0.676 | 0.512 | 0.537 |
| eegnet open | 0.563 | 0.578 | 0.605 | 0.644 | 0.525 | 0.537 |
| band power closed | 0.529 | 0.504 | 0.739 | 0.623 | 0.613 | 0.537 |
| band power open | 0.592 | 0.584 | 0.761 | 0.462 | 0.630 | 0.537 |

Sex alone barely predicts the BDI group and regressing it out changes little; STAI is about as
predictable as BDI. Sex-stratified results are noisy (11 high-BDI men).

**Reading.** The features are informative (C8) but BDI is essentially not decodable from them,
nor from band power (C9, C10), and not because of sex (C11). The seed vs rounding comparison is
therefore best reported on the positive control as well as on BDI: the conclusion is the same.
