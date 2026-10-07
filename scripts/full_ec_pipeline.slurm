#!/bin/bash
#SBATCH --job-name=eeg-fuzzy-ec-all
#SBATCH --account=rrg-glatard_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ec-all-%j.log
set -euo pipefail
base=${UNTRAINED_EEG_PROJECT:-"${SLURM_SUBMIT_DIR:-$PWD}"}
source "${EC_CONFIG:-$base/configs/ec_experiment.env}"
prepared=${PREPARED_EXPERIMENT:-"$base/experiments/ec_all_prepared"}
participants="$base/data/ec_all/participants.tsv"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
experiment="$base/experiments/bdi_job_${SLURM_JOB_ID}"
mkdir "$experiment"
cp "$prepared/manifest.json" "$experiment/"
# Share immutable cached files by hard link instead of duplicating them on disk.
ln "$prepared/inputs.npy" "$prepared/weights.pt" "$prepared/rows.npz" "$experiment/"
inside="/work/experiments/bdi_job_${SLURM_JOB_ID}"
for reference in reference reference_repeat; do
    bash "$base/scripts/container.sh" /work/scripts/02_extract_features.py run \
        --experiment "$inside" --mode rn --reference-name "$reference"
done
for ((rep=1; rep<=SR_REPETITIONS; rep++)); do
    bash "$base/scripts/container.sh" /work/scripts/02_extract_features.py run \
        --experiment "$inside" --mode sr --repetition "$rep" --perturbation-seed "$((SR_SEED_BASE+rep))"
done
bash "$base/scripts/container.sh" /work/scripts/04_compare_features.py --experiment "$inside" --repetitions "$SR_REPETITIONS"
normal="$base/.venv-normal/bin/python"
"$normal" "$base/scripts/03_probe_bdi.py" fit --features "$experiment/reference.npz" \
    --participants "$participants" --bundle "$experiment/fixed_probe.joblib" \
    --cond EC --folds "$CV_FOLDS" --cv-repeats "$CV_REPEATS" \
    --low-max "$BDI_LOW_MAX" --high-min "$BDI_HIGH_MIN" --C "$PROBE_C" \
    --out "$experiment/reference_probe.npz"
# Applying the repeat reference must also give identical classifier predictions.
"$normal" "$base/scripts/03_probe_bdi.py" apply --features "$experiment/reference_repeat.npz" \
    --participants "$participants" --bundle "$experiment/fixed_probe.joblib" \
    --out "$experiment/reference_repeat_probe.npz"
"$normal" - "$experiment" <<'PY'
from pathlib import Path
import sys,numpy as np
base=Path(sys.argv[1])
with np.load(base/'reference_probe.npz') as a, np.load(base/'reference_repeat_probe.npz') as b:
    assert np.array_equal(a['proba'],b['proba']), 'Reference classifier predictions differ'
PY
probe_files=()
for ((rep=1; rep<=SR_REPETITIONS; rep++)); do
    "$normal" "$base/scripts/03_probe_bdi.py" apply --features "$experiment/sr_rep-${rep}.npz" \
        --participants "$participants" --bundle "$experiment/fixed_probe.joblib" \
        --out "$experiment/sr_probe-${rep}.npz"
    probe_files+=("$experiment/sr_probe-${rep}.npz")
done
"$normal" "$base/scripts/05_compare_probes.py" --reference "$experiment/reference_probe.npz" \
    --runs "${probe_files[@]}" \
    --out "$experiment/probe_comparison"
