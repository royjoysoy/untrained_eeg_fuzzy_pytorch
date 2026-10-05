#!/bin/bash
# Preprocess all participants once (goal 1) and cache them in data/cache/.
#   sbatch slurm/prepare_cache.sh        (from the repo root)
#SBATCH --account=def-glatard   # change to your allocation
#SBATCH --job-name=eeg-cache
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
: "${UNTRAINED_EEG_DATA:?export UNTRAINED_EEG_DATA=/path/to/ds003478 before sbatch}"

module load python/3.11
source "${VENV:-$HOME/venv/untrained_eeg}/bin/activate"
python scripts/01_run_untrained.py
