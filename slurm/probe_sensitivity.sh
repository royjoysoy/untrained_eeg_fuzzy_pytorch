#!/bin/bash
# Probe sensitivity (nested C or PCA) on one dataset's saved phase 1 features.
#   CONFIG=configs/ds004504.yaml PROBE=pca sbatch slurm/probe_sensitivity.sh
#   CONFIG=configs/ds004504.yaml PROBE=nested sbatch --time=24:00:00 slurm/probe_sensitivity.sh
#SBATCH --account=def-glatard   # change to your allocation
#SBATCH --job-name=probe-sensitivity
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail

module load python/3.11
source "${VENV:-$HOME/venv/untrained_eeg}/bin/activate"
export OMP_NUM_THREADS=1  # one probe per worker
python scripts/probe_sensitivity.py --config "${CONFIG:-configs/ds003478.yaml}" --probe "${PROBE:-pca}"
