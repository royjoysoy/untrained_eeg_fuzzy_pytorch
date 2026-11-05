#!/bin/bash
# Phase 1 features of the large foundation models (REVE) on a GPU; too slow on CPUs.
#   CONFIG=configs/lemon.yaml sbatch slurm/phase1_gpu.sh          (from the repo root)
# Then rerun the probe on CPUs (it probes every extractor): python scripts/phase1_probe.py
#SBATCH --account=def-glatard   # change to your allocation
#SBATCH --job-name=phase1_gpu
#SBATCH --time=03:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail

module load python/3.11
source "${VENV:-$HOME/venv/untrained_eeg}/bin/activate"
export HF_HOME="$PWD/data/pretrained/hf" HF_HUB_OFFLINE=1  # foundation-model weights, no internet here
python scripts/phase1_features.py --config "${CONFIG:-configs/ds003478.yaml}" \
    --extractors ${EXTRACTORS:-reve_base reve_large} --device cuda
