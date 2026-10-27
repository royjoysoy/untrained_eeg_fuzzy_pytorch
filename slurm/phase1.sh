#!/bin/bash
# Phase 1: untrained vs pretrained models on one dataset (default ds003478).
#   bash scripts/00_download_pretrained.sh     (once, on the login node: needs internet)
#   sbatch slurm/phase1.sh                                   (from the repo root)
#   CONFIG=configs/ds004504.yaml sbatch slurm/phase1.sh      (another dataset)
# The first run also preprocesses the EEG (two inputs, cached in data/cache/).
# Resubmit if it times out: finished instances are skipped.
#SBATCH --account=def-glatard   # change to your allocation
#SBATCH --job-name=phase1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail

module load python/3.11
source "${VENV:-$HOME/venv/untrained_eeg}/bin/activate"
export HF_HOME="$PWD/data/pretrained/hf" HF_HUB_OFFLINE=1  # foundation-model weights, no internet here
python scripts/phase1_features.py --config "${CONFIG:-configs/ds003478.yaml}"
python scripts/phase1_probe.py --config "${CONFIG:-configs/ds003478.yaml}"
