#!/bin/bash
# Regression of MMSE (ds004504) on every model's features, with its permutation test.
#   CONFIG=configs/ds004504.yaml ARGS="--permutations 10000" sbatch slurm/regression.sh   (after slurm/phase1.sh)
#SBATCH --account=def-glatard   # change to your allocation
#SBATCH --job-name=regression
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail

module load python/3.11
source "${VENV:-$HOME/venv/untrained_eeg}/bin/activate"
export OMP_NUM_THREADS=1  # parallel permutations, one thread each
python scripts/regression.py --config "${CONFIG:-configs/ds004504.yaml}" ${ARGS:-}
