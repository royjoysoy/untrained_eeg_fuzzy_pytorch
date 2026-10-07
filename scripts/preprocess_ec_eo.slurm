#!/bin/bash
#SBATCH --job-name=eeg-preprocess-ec-eo
#SBATCH --account=rrg-glatard_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --output=logs/preprocess-ec-eo-%j.log
set -euo pipefail
base=${UNTRAINED_EEG_PROJECT:-"${SLURM_SUBMIT_DIR:-$PWD}"}
source "${EC_CONFIG:-$base/configs/ec_experiment.env}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$base/.venv-normal/bin/python" -u "$base/scripts/preprocess_ec_eo.py" \
    --bids-root "${UNTRAINED_EEG_DATA:?Set UNTRAINED_EEG_DATA to the ds003478 dataset folder}" \
    --out "$PREPROCESSED_ROOT"
