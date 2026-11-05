#!/bin/bash
# Phase 2: ensembles of K instances on one dataset (after slurm/phase1.sh).
#   sbatch slurm/phase2.sh                                   (from the repo root)
#   CONFIG=configs/ds004504.yaml sbatch slurm/phase2.sh      (another dataset)
#SBATCH --account=def-glatard   # change to your allocation
#SBATCH --job-name=phase2
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail

module load python/3.11
source "${VENV:-$HOME/venv/untrained_eeg}/bin/activate"
export OMP_NUM_THREADS=1  # parallel probes, one thread each
python scripts/phase2_ensembles.py --config "${CONFIG:-configs/ds003478.yaml}" --workers "$SLURM_CPUS_PER_TASK"
