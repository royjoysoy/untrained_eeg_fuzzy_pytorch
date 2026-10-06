# Controlled numerical variability of an untrained EEG CNN

This is a 1D EEG adaptation of an untrained CNN feature extractor. It does not
train the CNN. A seed-0 checkpoint is created once under ordinary rounding and
then reused byte-for-byte. The optional BDI classifier is a separate supervised
model fitted only on reference features.

## Prepared real-data experiment

Two subjects, sub-001 and sub-002, have been unpacked from the provided ZIPs.
Each raw recording has 66 channels. This experiment retains 64 electrodes and
excludes HEOG and VEOG. The 64-electrode choice includes M1, M2, CB1, and CB2;
these can instead be excluded by preparing a separate 60-electrode experiment.
Electrode names are validated and put into a canonical order for both subjects.

Run-01 is filtered at 1–40 Hz, average referenced across the selected electrodes,
and resampled to 250 Hz. Six 60-second blocks per subject produce 12 blocks of
shape (64, 15000). Eyes-open/closed blocks are preserved separately. No ICA or
artifact rejection is applied; numerical stability is distinct from EEG quality.

The provided sub-002 ZIP lacked its run-01 .set header. Its .fdt file also did
not match the public header's sample count. A matching .set/.fdt pair was retrieved
from the dataset's public OpenNeuro S3 storage. The original ZIP and .fdt version
are preserved. Event onsets were verified against the .set annotations for both
subjects. See ../data/repair_provenance.json for URLs and hashes.

## Submit exactly three perturbed runs

```bash
sbatch /home/mina94/fuzzy-pytorch/uncnn/untrained_eeg_fuzzy_pytorch/scripts/three_runs.slurm
```

The prepared artifacts are under experiments/real_eeg64/. The job copies them to
a fresh experiments/job_JOBID/ folder, runs two ordinary-rounding references and
exactly three perturbed repetitions (seeds 1001, 1002, 1003), then compares features.
Two identical references test for unwanted variability. All five executions use
the same compute node, one CPU thread, batch size 1, float32, model checkpoint,
inputs, and subject/block ordering. Initial job resources: 12 GiB and 3 hours;
full-data runtime needs measurement and may require adjusting the time request.

Outputs:

- reference.npz and reference_repeat.npz
- sr_rep-1.npz, sr_rep-2.npz, sr_rep-3.npz
- block_variability.csv: variation for each subject and block
- subject_variability.csv: variation in subject features averaged by condition
- summary.json: repeatability checks and overall feature variation
- logs/fuzzy3-JOBID.log (under the project root)

The manifests and outputs record input/weight/code/image hashes, configuration,
versions, CPU identity, and perturbation seeds. Altered controls, duplicate seeds,
nonfinite features, nonidentical references, and output overwrites are errors.

## What is perturbed

This image uses PRISM stochastic rounding, not legacy full MCA. Precision is
24 bits for binary32 and 53 bits for binary64. Only model(x) runs in stochastic
rounding. This includes convolutions, GroupNorm, activations, average pooling,
and PyTorch mean/std/covariance readout. Inputs are constructed once under ordinary
rounding and cached. Initialization and all calculations outside forward use
round-to-nearest; other Verificarlo backends stay IEEE.

The eeg variant produces 3904 features; faithful produces 9664. Each electrode
contributes robust-scaled, median-filtered, and absolute-derivative signals, so
64 electrodes become 192 input channels. Covariance readout excludes the diagonal.
These are 1D adaptations, not identical numerical replicas of a 3D MRI model.
The model's reported receptive field is the local convolution/pooling calculation;
GroupNorm and global feature readout also depend on the whole time segment.

## Dependency separation

.venv-normal/ contains ordinary preprocessing and classifier packages. Direct
versions are in requirements-normal.txt; the installed environment is recorded
in requirements-normal.lock.txt. It does not install or replace PyTorch.

The container's NumPy/SciPy files were incompatible. runtime-dependencies/ holds
clean ordinary NumPy 1.26.4 and SciPy 1.13.1, installed with --no-deps. The launcher
selects these packages while keeping the image's instrumented PyTorch. The SIF
and host packages are unchanged. test-dependencies/ holds the isolated pytest
runner. Do not install the normal requirements into the Fuzzy image.

## Prepare additional subjects or a different electrode choice

Run ordinary preprocessing once from the project directory, using a fresh output
folder. Add subjects explicitly (invalid sub-038 is excluded):

```bash
.venv-normal/bin/python scripts/01_preprocess.py \
  --bids-root /home/mina94/fuzzy-pytorch/uncnn/data \
  --out data/new_derivatives --channels 64 --subjects sub-001 sub-002
```

From a compute allocation, create a new fixed experiment:

```bash
bash scripts/container.sh /work/scripts/02_extract_features.py prepare \
  --deriv /work/data/new_derivatives --experiment /work/experiments/new_experiment \
  --variant eeg --seed 0 --dtype float32 --cond both
```

Change the prepared-experiment path in a copy of three_runs.slurm to use that
new experiment. Never overwrite a prepared checkpoint or cache. The saved
original derivative files need not be reread during repeated inference.

## Fixed downstream BDI classifier, when sufficient labeled subjects exist

The two downloaded subjects both have low BDI; they cannot support a two-class
classifier. Feature variability can still be measured. Default 5-fold CV requires
at least five eligible subjects per class. This compares low versus high BDI,
not a clinical diagnosis.

For a larger cohort, fit fold-specific scalers/classifiers only on reference
features, then apply the same bundle to each perturbed feature file:

```bash
.venv-normal/bin/python scripts/03_probe_bdi.py fit \
  --features experiments/job_JOBID/reference.npz \
  --participants ../data/participants.tsv --cond EC \
  --bundle experiments/job_JOBID/fixed_probe.joblib \
  --out experiments/job_JOBID/reference_probe.npz

.venv-normal/bin/python scripts/03_probe_bdi.py apply \
  --features experiments/job_JOBID/sr_rep-1.npz \
  --participants ../data/participants.tsv \
  --bundle experiments/job_JOBID/fixed_probe.joblib \
  --out experiments/job_JOBID/sr_probe-1.npz
```

Repeat apply for runs 2 and 3, then:

```bash
.venv-normal/bin/python scripts/05_compare_probes.py \
  --reference experiments/job_JOBID/reference_probe.npz \
  --runs experiments/job_JOBID/sr_probe-1.npz experiments/job_JOBID/sr_probe-2.npz experiments/job_JOBID/sr_probe-3.npz \
  --out experiments/job_JOBID/probe_comparison
```

Subjects are grouped before cross-validation, and scalers are fitted only on
training subjects in each fold. Applying perturbed features does not refit any
scaler/classifier or change splits. BLAS threads are limited to one. Comparisons
pair the same subject and fold repeat; they report probability differences and
classification flips. Classifier convergence failures stop execution.

## Tests

Ordinary preprocessing/probe checks:

```bash
.venv-normal/bin/python -m pytest -q tests_normal
```

CNN checks, from a compute allocation:

```bash
bash scripts/container.sh -c 'import sys; sys.path.insert(0,"/work/test-dependencies"); import pytest; raise SystemExit(pytest.main(["-q","/work/tests"]))'
```

The actual-data validation uses the saved real checkpoint and the first 256 time
points of one real EEG block. It checks repeatable references, observed feature
perturbations, and unchanged weights. It does not establish predictive accuracy.
