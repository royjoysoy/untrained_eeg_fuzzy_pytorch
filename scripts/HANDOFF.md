# Run this experiment with another person's EEG data

This code extracts fixed untrained CNN features and compares exactly three
PRISM stochastic-rounding repetitions against two ordinary-rounding references.
It does not train the CNN. The SIF implements PRISM SR, not legacy full MCA.

## Files to share

The accompanying eeg_fuzzy_handoff.tar.gz includes source, scripts, requirements,
tests and documentation. Share the container separately:

fuzzy-v2.6.0-pytorch2.2.1-avx2.sif
SHA256: 3f087a44c14954f6db4c5e4b16ecb753a40a3fb54b909ec4a4eabaf84ae0120d

Do not copy your virtual environment, cached inputs, old experiment manifests,
results or subject data into her experiment. She prepares fresh inputs and weights.
If weights must be shared across separate studies, use the optional --checkpoint
argument with this architecture's state_dict instead of generating different weights.

Required for feature variability:

- uncnn1d/: model.py, inputs.py, controls.py, __init__.py
- scripts/01_preprocess.py: dataset-specific preprocessing
- scripts/02_extract_features.py: preparation and controlled inference
- scripts/04_compare_features.py: comparisons and checks
- scripts/container.sh: image and environment launcher
- scripts/prepare_real.slurm: preprocessing and preparation job template
- scripts/three_runs.slurm: full three-repetition job template
- requirements-normal.txt and requirements-container.txt

Optional BDI prediction: scripts/03_probe_bdi.py and scripts/05_compare_probes.py.

## Data compatibility: check before running

01_preprocess.py is specifically written for OpenNeuro ds003478. It expects:

BIDS_ROOT/sub-XXX/eeg/sub-XXX_task-Rest_run-01_eeg.set
BIDS_ROOT/sub-XXX/eeg/sub-XXX_task-Rest_run-01_eeg.fdt
BIDS_ROOT/sub-XXX/eeg/sub-XXX_task-Rest_run-01_events.tsv

It expects the dataset's electrode names and event codes (1/3/5 closed; 2/4/6
open; 11-16 are repeated markers), and extracts six nonoverlapping 60-second
blocks. It also excludes this dataset's invalid sub-038.

For different EEG formats, montage, task, duration, or event naming, adapt
01_preprocess.py first. Do not assume this preprocessing transfers unchanged.
The chosen channels must be the same, in the same order, for every subject.
The current preparation accepts 60 or 64 electrodes. A different electrode
count requires a deliberate adaptation of its validation rules and a new model
checkpoint; the CNN itself supports configurable input-channel counts.

A custom preprocessing adapter should produce one sub-XXX.npz per subject with:

X: finite float32 (n_blocks, n_electrodes, n_time_points)
ch_names: unique electrode names in a consistent order
sfreq: sampling frequency in Hz
block: unique integers 1..6
cond: EC for odd block codes; EO for even block codes

All subjects must share channel order, sampling rate and segment length. Retaining
the original EC/EO block conventions is required by the current preparation;
adapt those checks as well if the study has another task or block design.
Preprocess only once and cache inputs once before all repetitions.

## Installation

Extract the archive and enter untrained_eeg_fuzzy_pytorch/. Set up an ordinary
Python environment (never install these requirements inside the Fuzzy image):

```bash
python3 -m venv .venv-normal
.venv-normal/bin/python -m pip install -r requirements-normal.txt
```

Put the SIF somewhere readable and set its path in scripts/container.sh. Update
its Apptainer module line for her cluster (or remove module load if apptainer is
already available). The supplied image requires an AVX2-capable x86-64 CPU and
contains Python 3.12; dependencies used inside it must be installed with its Python.

Create the local ordinary NumPy/SciPy directory using the image:

```bash
module load apptainer/1.3.5
PROJECT_DIR=$(pwd)
FUZZY_SIF=/absolute/path/fuzzy-v2.6.0-pytorch2.2.1-avx2.sif
apptainer exec --cleanenv --containall --no-mount hostfs \
  --bind "$PROJECT_DIR:/work" "$FUZZY_SIF" \
  python3 -m pip install --no-deps --target /work/runtime-dependencies \
  -r /work/requirements-container.txt
```

Do not install torch or replacement numerical dependencies into that directory.
The launcher deliberately uses ordinary NumPy/SciPy and the image's instrumented
PyTorch. Verify the versions before running on the new machine.

## Adapt both Slurm templates before preparing

In scripts/prepare_real.slurm change:

- #SBATCH --account to her allocation
- #SBATCH --output to her absolute log path
- base to her absolute project folder
- --bids-root to her dataset folder
- --subjects to the selected subject IDs (or omit to use all subject directories)
- --channels to 64 or 60 according to the agreed electrode policy

In scripts/three_runs.slurm change:

- #SBATCH --account and --output
- base to her absolute project folder
- the prepared experiment folder if it differs from experiments/real_eeg64
- memory/time as needed for cohort size; our two-subject check took 55 minutes

Also set the correct image path/module in scripts/container.sh BEFORE preparation,
because launcher and code hashes are fixed in the prepared manifest. Paths and
account settings in the Slurm job templates are not inside that code manifest.

Create logs/ BEFORE submitting, because Slurm must open its output file:

```bash
mkdir -p logs
sbatch scripts/prepare_real.slurm
```

Wait for successful preparation. Confirm experiments/real_eeg64/manifest.json,
inputs.npy, weights.pt and rows.npz exist. For a different configuration use new
preprocessing and experiment folders; overwrite protection is intentional.

Then submit:

```bash
sbatch scripts/three_runs.slurm
```

This uses one node and one CPU thread for all runs. Perturbation seeds are
1001/1002/1003. Weight seed stays 0, batch size stays 1, dtype stays float32.
SR applies to the full model forward including GroupNorm and feature readout;
ordinary rounding is restored outside forward. Other VFC backends stay IEEE.

## Evaluate and verify

Read experiments/job_JOBID/summary.json. A valid completed check reports:

reference_repeatable: true
all_controls_identical: true
n_runs: 3

Also inspect block_variability.csv and subject_variability.csv. Finite outputs,
fixed hashes, identical references and distinct perturbation seeds are checked
automatically. Slurm should report COMPLETED with exit code 0.

These quantify feature variability; classification accuracy needs a separately
validated labeled predictor. For BDI, the optional probe fits scalers/classifiers
on reference features once, saves them, and applies them unchanged to all runs.
It requires both label classes and at least five subjects per class for default
5-fold CV. Its participant column names and BDI thresholds are dataset-specific.

A three-run experiment is a pilot, not a precise estimate of uncertainty.
