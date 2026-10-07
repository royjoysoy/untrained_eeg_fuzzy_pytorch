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
