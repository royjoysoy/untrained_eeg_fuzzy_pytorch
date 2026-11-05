# Optimizing the untrained 1D CNN (phase-1 `cnn1d`)

Goal: find out which architecture and initialization choices make the frozen random-weight
`cnn1d` better or worse, on every phase-1 task, against handcrafted EEG features and the
phase-1 references (ShallowFBCSPNet, scattering, foundation models) and MiniRocket
(random convolution kernels, the standard time-series baseline). Status: code tested on
synthetic data; no real-data results yet.

## What stays fixed (ground rules)

- Phase-1 data, windows and probe: the "standard" cache, one feature vector per block (mean of
  its windows), `untrained_eeg.tasks.task_data` and `tasks.probe` (C = 1, `cv_seed` from the
  config). Our numbers are therefore directly comparable with `results/phase1/*/metrics.csv`.
  `tests/test_evaluate.py` checks our generic probe equals `tasks.probe` for all four task types.
- Subject-wise CV: `tasks._outer_splits` (participant-level, grouped, or within-subject).
- `ArchConfig()` = phase-1 `cnn1d`, bit-identical weights and features (`tests/test_arch.py`).
- One factor at a time (`arch.ablation_grid()`, 31 configs), 30 weight seeds (as phase 1).
- Selection only inside training folds: (a) `nested` per task; (b) leave-one-dataset-out
  (`summarize.py`): a config is chosen on the other datasets' tasks, then reported on the
  held-out one. Descriptive per-config numbers (`results.csv`) are never the headline.
- Riemannian tangent space and MiniRocket's biases fitted inside training folds only.
- Outputs under `results/optimization/<M-D-YY>/` in your own clone; writers never overwrite.

## Ablation factors (from `cnn1d`: conv k=25, 32 filters -> ELU -> pool 4 -> conv k=11, 64 -> ELU -> mean)

| | factor | values |
|---|---|---|
| a | channel handling | all electrodes mixed (current), depthwise then 1x1 spatial, per electrode |
| b | input | raw (current), un-CNN 3 versions (robust, median-filtered, abs derivative) |
| c | first kernel / dilation | k1 = 7, 13, 25, 51, 101; dilation 2, 4 |
| d | multi-scale first layer | 7-25-51-101, 13-51 |
| e | depth / width | 1-4 blocks; 16, 32, 64, 128 filters |
| f | initialization | PyTorch default, Kaiming normal, Xavier, Gaussian; scale x0.1, x10; zero bias |
| g | nonlinearity / readout | ELU, ReLU, tanh, none; mean, +std, +PPV/max; all blocks; +covariance |

`cnn1d` has no normalization layer, so weight scale and input amplitude (cache is in microvolts)
can matter here, unlike in `uncnn1d`.

## Environment on Rorqual

```bash
module load python/3.12 scipy-stack
virtualenv --no-download $HOME/venv/uneeg_opt && source $HOME/venv/uneeg_opt/bin/activate
pip install --no-index torch scikit-learn pandas pyyaml joblib matplotlib
pip install pyriemann fooof aeon  # login node (needs internet)
python -m pytest -q optimization/tests
```

## Run order

```bash
S=/project/rrg-glatard/shared/untrained_eeg
OUT=$PWD/results/optimization/11-5-26
# Step 0 (interactive job, about 10 min): reproduce phase-1 cnn1d on ds003478
python optimization/extract.py cnn --dataset ds003478 --data-root $S/data --out $OUT --configs default --seeds 0-29
python optimization/step0_check.py --dataset ds003478 --data-root $S/data --phase1 $S/results/phase1 --out $OUT
# runtime estimate (interactive job)
python optimization/benchmark.py --data-root $S/data
# Steps 1-3 (after approval)
S=$S OUT=$OUT VENV=$HOME/venv/uneeg_opt bash optimization/slurm/run_optimization.sh
# random search over the 2-3 strongest factors, then extract + evaluate with --search-list
python optimization/random_search.py --space space.json --n 12 --out $OUT/search_configs.txt
```

## Files

| File | Role |
|---|---|
| `arch.py` | configurable `cnn1d` + ablation grid |
| `common.py` | datasets, phase-1 cache loading, block averaging, phase-1 file layout |
| `extract.py` | CNN (`cnn`) and handcrafted (`handcrafted`) block features |
| `handcrafted.py` | band power (abs/rel), PAF + theta/beta, aperiodic (FOOOF), Hjorth, spectral + permutation entropy, Riemannian covariances (broadband + 5 bands) |
| `evaluate.py` | c1 probe, generic probe (same splits), nested selection, tangent space |
| `evaluate_dataset.py` | `per_seed.csv`, `nested.csv` per dataset |
| `summarize.py` | `results.csv`, `lodo.csv`, `nested.csv`, `references.csv`, `headline.csv` |
| `report.py` | one figure per factor (all tasks, seed spread, reference lines) |
| `minirocket.py` | MiniRocket baseline (aeon): biases fitted on training windows of each outer split, same blocks, splits and probe; 10 seeds |
| `step0_check.py`, `benchmark.py`, `random_search.py`, `slurm/run_optimization.sh` | as named |
