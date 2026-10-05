#!/bin/bash
# Goal 3: one Fuzzy PyTorch (MCA) sample per array task.
#   sbatch slurm/mca_variability.sh      (from the repo root, after prepare_cache.sh)
# Array size should match n_mca_samples in the config (default 50 -> 0-49).
#SBATCH --account=def-glatard   # change to your allocation
#SBATCH --job-name=eeg-mca
#SBATCH --array=0-49
#SBATCH --time=02:00:00         # TODO(fuzzy-image): adjust once MCA slowdown is known
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=slurm/logs/%x-%A_%a.out
set -euo pipefail
: "${UNTRAINED_EEG_DATA:?export UNTRAINED_EEG_DATA=/path/to/ds003478 before sbatch}"
# TODO(fuzzy-image): Roy will provide the image; see environment/fuzzy_container.md
: "${FUZZY_IMAGE:?export FUZZY_IMAGE=/path/to/fuzzy-pytorch.sif before sbatch}"

module load apptainer
apptainer exec \
    --bind "$PWD" --bind "$UNTRAINED_EEG_DATA" --pwd "$PWD" \
    --env PYTHONPATH="$HOME/fuzzy_pkgs" \
    --env OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" \
    "$FUZZY_IMAGE" \
    python3 scripts/03_mca_variability.py --sample "$SLURM_ARRAY_TASK_ID"
