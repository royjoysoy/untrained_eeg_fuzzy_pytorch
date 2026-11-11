#!/bin/bash
# Step (b), before the main run: build the raw-based caches (0.5-45 Hz), then step 0 on every
# dataset: the reference level re-extracted for seeds 0-2 of both architectures (and the
# handcrafted features) must equal step (a) bit for bit, and every level must keep the
# reference's blocks. Run on the LOGIN node:
#
#   export S=/project/rrg-glatard/shared/untrained_eeg
#   export A=/project/rrg-glatard/royjoy/uneeg_opt_results/optimization   # step (a) results
#   export C=/project/rrg-glatard/royjoy/uneeg_prep_cache                 # new caches
#   export OUT=$PWD/results/optimization/11-9-26
#   export VENV=$HOME/venv/uneeg_opt
#   bash optimization/slurm/run_prep_step0.sh
# Then: grep "STEP 0" logs/prep-step0-*.log   (every dataset must say MATCH)
set -euo pipefail
: "${S:?}" "${A:?}" "${C:?}" "${OUT:?}" "${VENV:?}"
ACCOUNT=${ACCOUNT:-rrg-glatard_cpu}
DATASETS=${DATASETS:-"ds003478 ds002778 ds003490 ds004504 lemon sleep_edf bnci2014_001"}
COMBINED="cov_channels=32,n_filters=256,readout=meanstd_ppv_max"
PY="export PYTHONUNBUFFERED=1 && source $VENV/bin/activate && python"
mkdir -p logs "$OUT" "$C"
for ds in $DATASETS; do
  mem=16G; [ "$ds" = lemon ] && mem=80G
  common="--dataset $ds --data-root $S/data --cache-root $C"
  b=$(sbatch --parsable --account=$ACCOUNT --job-name=prep-build-$ds --cpus-per-task=1 --mem=$mem \
      --time=06:00:00 --output=logs/prep-build-$ds-%j.log \
      --wrap="$PY optimization/prep.py build $common")
  z=$(sbatch --parsable --account=$ACCOUNT --job-name=prep-step0-$ds --dependency=afterok:$b \
      --cpus-per-task=8 --mem=$mem --time=06:00:00 --output=logs/prep-step0-$ds-%j.log \
      --wrap="export SLURM_CPUS_PER_TASK=1; \
              $PY optimization/extract.py cnn $common --out $OUT --prep ref --configs default $COMBINED --seeds 0-2 && \
              $PY optimization/extract.py handcrafted $common --out $OUT --prep ref --jobs 8 && \
              $PY optimization/prep_step0.py $common --out $OUT --step-a $A/11-5-26 $A/11-5-26b --seeds 0-2 --check-levels")
  echo "prep_build $ds $b" >> "$OUT/job_ids_step0.txt"; echo "prep_step0 $ds $z" >> "$OUT/job_ids_step0.txt"
  echo "$ds: build=$b step0=$z"
done
