#!/bin/bash
# Goal 2: all models x conditions x n_seeds (about 15 min on 8 CPUs).
#   sbatch slurm/seed_variability.sh     (from the repo root, after prepare_cache.sh)
#SBATCH --account=def-glatard   # change to your allocation
#SBATCH --job-name=eeg-seeds
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
: "${UNTRAINED_EEG_DATA:?export UNTRAINED_EEG_DATA=/path/to/ds003478 before sbatch}"

module load python/3.11
source "${VENV:-$HOME/venv/untrained_eeg}/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python scripts/02_seed_variability.py
