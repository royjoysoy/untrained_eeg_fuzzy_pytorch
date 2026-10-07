# Frozen untrained CNN: controlled EC experiment

This workflow uses 64 EEG electrodes, 60-second blocks, an untrained frozen 1D
CNN, two reference passes and five PRISM stochastic-rounding runs. A separate
logistic-regression classifier is fitted on reference features and reused for
perturbed features. No CNN training occurs. Settings are in
`configs/ec_experiment.env`; defaults are low BDI <=7 and high BDI >=13.
These settings differ from the broader prototype in the root README.

## Files

| File | Role |
| --- | --- |
| `preprocess_ec_eo.py` | Independently validate and save EC/EO blocks from ds003478 |
| `01_preprocess.py` | Shared channel constants and older preprocessing entry point |
| `02_extract_features.py` | Cache fixed inputs/weights; reference or perturbed inference |
| `03_probe_bdi.py` | Reference-only fold fitting; fixed-model prediction |
| `04_compare_features.py` | Verify references and controls; summarize feature variation |
| `05_compare_probes.py` | Compare probabilities, class decisions and metrics |
| `container.sh` | Isolated container launch with early PRISM seed |
| `preprocess_ec_eo.slurm` | Optional raw-data preprocessing job |
| `prepare_ec_all.slurm` | Prepare the existing EC data once |
| `full_ec_pipeline.slurm` | Complete reference/SR/classification experiment |

All scripts need the root-level `uncnn1d/` package. Install requirements from
`requirements-normal.txt` and `requirements-container.txt`, not the broader
prototype's `environment/requirements.txt`.

## Install once

From the repository root:

```bash
python3 -m venv .venv-normal
.venv-normal/bin/python -m pip install -r requirements-normal.txt
mkdir -p logs data experiments
```

Supply the image separately: `fuzzy-v2.6.0-pytorch2.2.1-avx2.sif`.
Its SHA256 is
`3f087a44c14954f6db4c5e4b16ecb753a40a3fb54b909ec4a4eabaf84ae0120d`.
It requires an AVX2-capable x86-64 CPU and uses Python 3.12. Set its absolute
path with `FUZZY_IMAGE`; do not install replacement PyTorch.

```bash
export FUZZY_IMAGE=/absolute/path/fuzzy-v2.6.0-pytorch2.2.1-avx2.sif
module load apptainer/1.3.5  # adapt for your cluster
apptainer exec --cleanenv --containall --no-mount hostfs \
  --bind "$PWD:/work" "$FUZZY_IMAGE" \
  python3 -m pip install --no-deps --target /work/runtime-dependencies \
  -r /work/requirements-container.txt
```

The launcher uses ordinary local NumPy/SciPy alongside the image's instrumented
PyTorch. If your cluster uses a different module name, set `APPTAINER_MODULE`.
Outside a module-enabled cluster, Apptainer must already be on PATH.

## Input data

For existing preprocessed EC data, use one `sub-XXX.npz` per subject with:

- `X`: finite float32 `(number_of_blocks, 64, 15000)`.
- `block`: unique numbers from 1, 3, 5.
- `cond`: `EC` for every block.
- `ch_names`: the same ordered 64 names for all subjects.
- `sfreq`: 250 Hz.

One to three valid blocks per subject are accepted. Features are averaged
within each subject; subjects are not combined into a class-average signal.
`participants.tsv` needs unique `participant_id` values matching filenames and
numeric `BDI` values. Five-fold classification needs at least five subjects in
each retained class. Different clinical labels need a deliberate code adaptation.

For raw ds003478 data only, first set `UNTRAINED_EEG_DATA` to the dataset root
containing `sub-*` directories and `participants.tsv`, then submit:

```bash
export UNTRAINED_EEG_DATA=/absolute/path/ds003478
sbatch --account=YOUR_ACCOUNT scripts/preprocess_ec_eo.slurm
```

The output defaults to `data/preprocessed_ec_eo/{EC,EO}`. Numeric event codes are
accepted even when text labels say STATUS. Each 60-second block is checked
independently for marker coverage, recording bounds, condition transitions and
discontinuities. Only the dataset's invalid sub-038 is automatically excluded.
A missing EO block does not discard valid EC blocks. This is minimal
preprocessing: 1–40 Hz, average reference, 250 Hz, no ICA or full artifact cleaning.
Different datasets need an adapter for their event codes, montage and file format.

## Submit the EC experiment

Skip raw preprocessing when compatible EC derivatives already exist.
Configure paths through environment variables or `configs/ec_experiment.env`:

```bash
export EC_DATA_DIR=/absolute/path/preprocessed_ec_eo/EC
export PARTICIPANTS="$EC_DATA_DIR/participants.tsv"
prep_job=$(sbatch --parsable --account=YOUR_ACCOUNT scripts/prepare_ec_all.slurm)
echo "Preparation job: $prep_job"
sbatch --account=YOUR_ACCOUNT --dependency=afterok:"$prep_job" scripts/full_ec_pipeline.slurm
```

Submit from the repository root: relative log paths require `logs/` to exist.
The templates default to the Rorqual account `rrg-glatard_cpu`; override it for
another allocation. Preparation requests 24 GB/two hours. The full job requests
12 GB/48 hours and keeps all passes on one node/CPU. Adjust resources for your
cohort. Hard-link staging requires inputs and the project on the same filesystem;
use `cp` instead of `cp -l` when they are on different filesystems.
Participant labels are copied into the staging directory as a fixed snapshot.

Preparation saves immutable inputs, row order, weights and their manifest under
`experiments/ec_all_prepared`. Fresh output directories are required. Do not edit
hard-linked input files or protected code after preparation. A new configuration
needs a new prepared experiment, not an overwritten manifest.

The full job performs both reference passes, all stochastic passes, feature
comparison, reference classifier fitting and fixed-classifier comparisons.
`SR_REPETITIONS` controls both extraction and every comparison stage, default 5.
If increasing it, increase the full job's time request appropriately.

## Controls and outputs

Weights use seed 0 and remain frozen, in evaluation mode and without gradients.
The checkpoint and cached inputs are checked by SHA256. Defaults are float32,
batch size 1, deterministic PyTorch algorithms and one CPU thread. PRISM
precision is binary32 24 bits/binary64 53 bits. Seeds 1001–1005 are supplied at
process startup, before PyTorch loads; the runtime setter alone does not reset
already initialized streams. Perturbation applies only around the full CNN
forward, including normalization/pooling/readout.

The reference classifier uses subject-level five-fold CV repeated ten times.
Scalers fit training subjects only. Fifty reference fold classifiers are saved
and reused without fitting on perturbed features. Data membership, labels and
classifier identity are verified.

Results go to `experiments/bdi_job_JOBID`. Read `summary.json` and
`probe_comparison/summary.json`; reference repeatability and identical controls
must pass. CSVs contain per-block, per-subject feature and prediction variability.
Logs are `logs/ec-cache-JOBID.log` and `logs/ec-all-JOBID.log`. Subject block counts
may differ; keep them fixed across repetitions and report them.

## Tests

```bash
.venv-normal/bin/python -m pytest -q tests_normal
```

For model tests, install only pytest's test dependencies locally using container
Python (without replacing PyTorch):

```bash
apptainer exec --cleanenv --bind "$PWD:/work" "$FUZZY_IMAGE" \
  python3 -m pip install --no-deps --target /work/test-dependencies \
  pytest==8.3.3 pluggy==1.5.0 iniconfig==2.0.0 packaging==24.2 pygments==2.18.0
bash scripts/container.sh -c 'import sys; sys.path.insert(0,"/work/test-dependencies"); import pytest; raise SystemExit(pytest.main(["-q","/work/tests"]))'
```

Data, images, environments, caches and results are excluded from Git. Older
`validate_real.py`/`validate_pipeline.py` scripts refer to historical local pilot
fixtures; they are not required for this portable submission workflow.
