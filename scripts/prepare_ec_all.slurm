#!/bin/bash
#SBATCH --job-name=eeg-ec-cache
#SBATCH --account=rrg-glatard_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ec-cache-%j.log
set -euo pipefail
base=${UNTRAINED_EEG_PROJECT:-"${SLURM_SUBMIT_DIR:-$PWD}"}
source "${EC_CONFIG:-$base/configs/ec_experiment.env}"
source_data=$EC_DATA_DIR
staged="$base/data/ec_all"
mkdir "$staged"
# Share existing files without duplicate storage. Treat both paths as read-only.
cp -l "$source_data"/sub-*.npz "$staged/"
# Snapshot labels so later edits to the source TSV cannot change this experiment.
cp "$PARTICIPANTS" "$staged/participants.tsv"
bash "$base/scripts/container.sh" /work/scripts/02_extract_features.py prepare \
    --deriv /work/data/ec_all --experiment /work/experiments/ec_all_prepared \
    --variant "$CNN_VARIANT" --seed "$WEIGHT_SEED" --dtype "$INPUT_DTYPE" --cond EC
