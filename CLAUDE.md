# CLAUDE.md

Context for AI assistants (and humans) working in this repo.

## Project
Brainhack Montreal Fall 2026, prototype, aiming at a short paper. Question: how do
untrained (random-weight) CNNs compare with trained (pretrained) models on EEG, across
several tasks on open datasets? Variability (across seeds, and numerical via Fuzzy PyTorch / MCA)
is part of the comparison. Lead: Roy Seo (@royjoysoy).
The repo is public and used by Brainhack beginners: keep it clean, short, and well commented.

## Design decisions (settled)
- Untrained models: small custom 1D CNN, ShallowFBCSPNet, EEGNet. Random weights, eval mode, never trained.
  Plus Mallat's wavelet scattering (Kymatio, `WaveletScattering` in `models.py`): fixed wavelets, no
  weights, one instance per largest scale J = 3..8.
  The two braindecode models are copied as plain PyTorch in `models.py` (no braindecode/MNE in the
  Fuzzy environment); `tests/test_models.py` checks they stay bit-identical to braindecode 1.8.1.
- Trained models: all publicly available pretrained checkpoints for scalp EEG (CNNs and foundation
  models such as LaBraM), including pretrained EEGNet and ShallowFBCSPNet next to their untrained
  versions. Same frozen features + probe for every model. We do not train our own models.
  Open weights only (no registration); CC BY-NC weights are fine. REVE's weights became open in
  2026-09 (REVE Responsible Use License), so REVE base and large are in (out on ds002778: leakage).
  Neuro-GPT (GPL-3.0, TUEG only): its encoder only, on its 22 TUH channels (missing ones interpolated).
  Pretrained EEGNet/ShallowFBCSPNet load into plain-PyTorch copies in `models.py` (no braindecode in
  Fuzzy); loaders in `pretrained.py`, checked bit-identical to braindecode 1.8.1 by
  `tests/test_pretrained.py`. Weights go in `data/pretrained/` (`scripts/00_download_pretrained.sh`).
  Foundation models (`foundation.py`: EEGPT, LaBraM, BENDR, CBraMod, CodeBrain, EEG-DINO, Signal-JEPA, REVE, Neuro-GPT)
  run through braindecode 1.8.1 itself, each with its pretraining input (rate, channels, window, scale).
- Run `tests/test_models.py` and `tests/test_pretrained.py` with `--noconftest` outside the
  container (`tests/conftest.py` is for the EC experiment and needs the Fuzzy container).
- Open datasets only (no registration-gated data). An evaluation dataset must not be in a
  checkpoint's pretraining data (leakage).
- Phase 1: single models on every task. Phase 2: ensembles (seeds, checkpoints, models).
- Fuzzy PyTorch / MCA is paused: variability is seeds, checkpoints and CV splits for now.
- Paper: no Fuzzy/MCA angle (left for later work). Primary significance test: the median instance's
  refit permutation (`scripts/permutation.py --ensemble --maxt`, statistic `median`), max-T
  corrected per task (each task's extractors are one family); ensemble, best instance and first
  instance are secondary. Probe C = 1 is primary; nested C and PCA (`probe_sensitivity.py`) are
  sensitivity analyses. BENDR is left out (`foundation.EXCLUDED`): we could not reproduce its input.
- Same seed ≠ same weights across PyTorch builds (last-bit differences, e.g. torch 2.14 vs Fuzzy 2.2):
  run the seed baseline with the same build as the Fuzzy runs (normal rounding), or reuse saved weights.
- Untrained and pretrained networks are compared the same way: frozen feature extractor (head removed)
  + downstream probe. Variability is measured on the probe, not on embeddings.
- Probe: standardized logistic regression, stratified 5-fold CV with a fixed split; per-task details
  in `tasks.py` (participant-grouped folds below participant level, within subject for motor imagery,
  multinomial for sleep staging). Code: `scripts/phase1_*.py`, `phase2_ensembles.py`, `stats.py`.
- Datasets (one config each in `configs/`): ds003478 (eyes closed vs open, BDI), ds004504 (AD, FTD),
  ds002778 (Parkinson's), MPI-LEMON (age, authors' preprocessed EC/EO), BNCI2014_001 (left vs right
  motor imagery), Sleep-EDF (5 stages, first 20 subjects, 30 epochs per stage and night).
- ds003478: v1.1.0, run-01 only, 1-minute eyes-open/closed blocks → 2 s windows, participant 544 excluded.
- ds002778 (Parkinson's): the patients' off-medication session. Confirmed.
- BDI cutoffs: ≤6 low / ≥17 high (equivalent to the original study's <7 / >16 for integer scores). Confirmed.
- License MIT.

## Open items (ask Roy, don't assume)
- Foundation models run through braindecode, not plain-PyTorch copies (the settled rule above was
  for the Fuzzy container, now paused): confirm, or port them when Fuzzy resumes.
- BENDR's input scaling (we assume per-channel min-max to [-1, 1]) is not documented: verify with
  the authors' code. (EEGPT's millivolts are confirmed by the authors' pretraining script.)
- MIRepNet and BrainOmni need braindecode's development version; ST-EEGFormer, BIOT, ZUNA, DIVER-1,
  NeuroRVQ, SleepFM: pretraining corpora still to check (`docs/pretrained_models.md`).
- ds002778: its README asks authors to email the dataset curators before submitting a manuscript.
- Fuzzy PyTorch image name/tag: placeholders marked `TODO(fuzzy-image)` in `environment/` and `slurm/`.
- Shared data path in Rorqual `/project` space.

## Data quirks (ds003478)
- Trigger codes 1–6 repeat every 500 ms within a 1-minute block; odd = eyes closed, even = eyes open, one code per block. Codes 11–16 are 2 s ticks (ignored), 17 = start/finish.
- Some blocks are incomplete (recording started late); blocks < 60 s are skipped.
- Some files have an extra EKG channel → we pick an explicit list of 60 scalp channels (config `channels`).
- 516 has no run-02 (irrelevant, we use run-01). 544: invalid participant, excluded.
- HEOG/VEOG may be swapped; some channels were interpolated by the authors; no raw data to revert to.
- Run-02 was never quality-checked by the dataset authors.
- After exclusions and cutoffs: 113 participants, 74 low, 39 high.

## Conventions
- No data in the repo. Each dataset is downloaded into `data/raw/<dataset>` the first time it is
  needed (`download.py`, resumable); `data_dir` or the dataset's variable (`data_env` in its config:
  `UNTRAINED_EEG_DATA` for ds003478, `UNTRAINED_EEG_DS004504`, ...) point to an existing copy
  instead, which is never written to. Never hardcode a data path.
- Never commit `data/` or `results/`.
- Preprocessing runs outside the Fuzzy container; scripts 02–04 read only the cache (`build=False`).
- One process = one MCA sample (`03_mca_variability.py --sample i`).
- Outside the container, MCA samples must be bit-identical (sanity check).
- Scripts put `src/` on `sys.path` themselves; there is no package install step.
- Commits must use an email verified on the `royjoysoy` GitHub account.
