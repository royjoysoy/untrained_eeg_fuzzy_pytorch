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
pip install pyriemann fooof "scipy==1.17.1"   # login node (needs internet); scipy as in phase 1
python -m pytest -q optimization/tests
# MiniRocket in its own environment: aeon pulls numba, which pins an older numpy, and the
# main environment must keep phase 1's numpy 2.4.2 (the probe must reproduce phase 1 exactly)
deactivate
virtualenv --no-download $HOME/venv/uneeg_mr && source $HOME/venv/uneeg_mr/bin/activate
pip install --no-index torch scikit-learn pandas pyyaml joblib && pip install aeon "scipy==1.17.1"
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
S=$S OUT=$OUT VENV=$HOME/venv/uneeg_opt VENV_MR=$HOME/venv/uneeg_mr bash optimization/slurm/run_optimization.sh
# random search over the 2-3 strongest factors, then extract + evaluate with --search-list
python optimization/random_search.py --space space.json --n 12 --out $OUT/search_configs.txt
```

## What gets recorded (for the manuscript)

| Where | What |
|---|---|
| `run_info/*.json` | every script run: git commit and local changes, command, Python and package versions, CPU model, usable CPUs, threads, host, SLURM job/array ids, start, end, wall seconds, status |
| `features/**/*.npz` (`meta`) | config, seed, feature dimension, receptive field (s), parameter count, windows, extraction seconds |
| `<dataset>/per_seed.csv`, `nested.csv`, `minirocket.csv` | per seed and task: AUC, balanced accuracy, n, probe / nested / MiniRocket seconds, configs chosen in nested selection |
| `summary/results.csv` | mean / SD / min / max over seeds, delta vs current cnn1d, ms per window, parameters |
| `summary/headline.csv`, `lodo.csv`, `nested.csv`, `references.csv` | headline comparison and the selection procedures |
| `job_ids.txt`, `summary/compute_cost.csv`, `compute_by_stage.csv` | SLURM accounting per job: state, wall time, used and allocated CPU-hours, peak memory; per stage and dataset; run soon after the jobs end |
| `logs/` | stdout/stderr of every job |

Times are on Rorqual CPUs with the threads shown in `run_info`; report hardware and threads
with any timing.

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
| `runinfo.py`, `compute_cost.py` | run records and the compute budget (sacct) |
| `step0_check.py`, `benchmark.py`, `random_search.py`, `slurm/run_optimization.sh` | as named |

## Step (b): preprocessing (architecture and probe fixed)

Question: does changing the input (not the network) close the gap to handcrafted features?
Two fixed architectures: the phase-1 `default` and step (a)'s LODO pick
`cov_channels=32,n_filters=256,readout=meanstd_ppv_max`. One factor at a time away from the
phase-1 cache ("ref"); 30 weight seeds; MiniRocket (10 seeds) and handcrafted features get the
same input. Levels, and where they apply, are in `prep.py`:

| factor | levels | applies to |
|---|---|---|
| window length | 4 s, 10 s (cached windows joined) | resting datasets; BNCI 4 s only; not Sleep-EDF |
| normalization | z-score per participant and channel; per window and channel | all |
| artifact windows | peak-to-peak > 150 uV; > median + 5 MAD (per participant) | resting datasets |
| block aggregation | median, 20 % trimmed mean | resting datasets |
| band-pass | extra 30 Hz low-pass (cache); 0.5-45 Hz (new cache from raw) | all |
| sampling rate | 256 Hz (phase-1 `fm256` cache); 256 Hz with first kernel and pooling x2 | not Sleep-EDF |

Not a factor: re-referencing (the cache is already average-referenced, so it changes nothing).
A block whose windows would all be rejected keeps them all, so every level has the same samples
and folds. Selection: `nested` (level chosen inside the outer training folds, per task and seed)
and leave-one-dataset-out (`prep_select.py`). Handcrafted features use whole blocks, so window
length and aggregation do not change them.

```bash
S=/project/rrg-glatard/shared/untrained_eeg
A=/project/rrg-glatard/royjoy/uneeg_opt_results/optimization       # step (a) results
C=/project/rrg-glatard/royjoy/uneeg_prep_cache                     # new caches (bp05_45)
OUT=$PWD/results/optimization/11-9-26
# 1. estimate from step (a)'s measured times; ask before running
python optimization/prep_estimate.py --step-a $A/11-5-26 $A/11-5-26b
# 2. build the 0.5-45 Hz caches from raw (needs MNE; one job per dataset)
python optimization/prep.py build --dataset ds003478 --data-root $S/data --cache-root $C
# 3. step 0: reference level = step (a), bit for bit; every level keeps the same blocks
python optimization/extract.py cnn --dataset ds003478 --data-root $S/data --out $OUT --prep ref \
    --configs default cov_channels=32,n_filters=256,readout=meanstd_ppv_max --seeds 0-2
python optimization/extract.py handcrafted --dataset ds003478 --data-root $S/data --out $OUT --prep ref --jobs 8
python optimization/prep_step0.py --dataset ds003478 --data-root $S/data --out $OUT --cache-root $C \
    --step-a $A/11-5-26 $A/11-5-26b --check-levels
# 4. everything else (after approval)
S=$S C=$C OUT=$OUT VENV=$HOME/venv/uneeg_opt VENV_MR=$HOME/venv/uneeg_mr bash optimization/slurm/run_prep.sh
```

| File | Role |
|---|---|
| `prep.py` | levels, where they apply, transforms of cached windows, raw-based cache build |
| `prep_step0.py` | reference = step (a) bit for bit; same blocks at every level |
| `prep_select.py` | nested and LODO selection of the level; `prep_summary/` tables |
| `prep_estimate.py` | CPU-hours from step (a)'s measured times |
| `slurm/run_prep.sh` | all jobs |

Outputs: `<out>/prep/<level>/<dataset>/` (features, `per_seed.csv`, `minirocket.csv`) and
`<out>/prep_summary/` (`results.csv`, `factor_effects.csv`, `nested.csv`, `lodo.csv`, `headline.csv`).
