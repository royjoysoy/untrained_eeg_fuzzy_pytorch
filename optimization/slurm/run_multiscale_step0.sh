#!/bin/bash
# Step (a''), before the main run: per dataset, extract
#   - the reference (seeds 0-2) -> must equal step (a')'s combined + Hjorth features bit for bit,
#   - every other level (seed 0) -> sanity checks on real data (multiscale_step0.py),
# then check, and on PROBE_DATASETS also time the probe of seed 0 at every level (for the
# CPU-hour estimate; sleep_edf is the large-dataset check). Run on the LOGIN node:
#
#   export S=/project/rrg-glatard/shared/untrained_eeg
#   export A=/project/rrg-glatard/royjoy/uneeg_opt_results/optimization   # steps (a), (b), (a')
#   export OUT=$PWD/results/optimization/11-13-26
#   export VENV=$HOME/venv/uneeg_opt
#   bash optimization/slurm/run_multiscale_step0.sh
# Then: grep "STEP 0" logs/multiscale-step0-*.log   (every dataset must say MATCH)
set -euo pipefail
: "${S:?}" "${A:?}" "${OUT:?}" "${VENV:?}"
ACCOUNT=${ACCOUNT:-rrg-glatard_cpu}
DATASETS=${DATASETS:-"ds003478 ds002778 ds003490 ds004504 lemon sleep_edf bnci2014_001"}
PROBE_DATASETS=${PROBE_DATASETS:-"ds004504 bnci2014_001 sleep_edf"}
BASE="n_filters=256,readout=hjorth"   # combined + Hjorth (optimization/multiscale.py BASE)
LEVELS="ms2 last3 ms3 fb fb_ms2"
PY="export PYTHONUNBUFFERED=1 && source $VENV/bin/activate && python"
# the main environment must match phase 1 (numpy 2.4.2, scipy 1.17.1, scikit-learn 1.8.0, torch 2.14.1)
bash -c "$PY -c 'import numpy,scipy,sklearn,torch;v=(numpy.__version__,scipy.__version__,sklearn.__version__,torch.__version__);print(\"main env:\",v);assert v[0].startswith(\"2.4.2\") and v[1].startswith(\"1.17.1\") and v[2].startswith(\"1.8.0\") and v[3].startswith(\"2.14.1\"),\"versions differ from phase 1\"'"
[ -d "$A/11-11-26/readout/hjorth" ] || { echo "step (a') results not found: $A/11-11-26/readout/hjorth"; exit 1; }
mkdir -p logs "$OUT"
[ -e "$OUT/job_ids_step0.txt" ] && { echo "$OUT/job_ids_step0.txt exists: already submitted (check squeue)"; exit 1; }
for ds in $DATASETS; do
  mem=16G; [ "$ds" = lemon ] && mem=80G
  common="--dataset $ds --data-root $S/data --out $OUT"
  # 1 CPU: extraction is single-threaded (as in every earlier step)
  e=$(sbatch --parsable --account=$ACCOUNT --job-name=multiscale-ext0-$ds --cpus-per-task=1 --mem=$mem \
      --time=12:00:00 --output=logs/multiscale-ext0-$ds-%j.log \
      --wrap="$PY optimization/extract.py cnn $common --multiscale ref --configs $BASE --seeds 0-2 && \
              for lv in $LEVELS; do $PY optimization/extract.py cnn $common --multiscale \$lv --configs $BASE --seeds 0 || exit 1; done")
  cpus=1; probe=""
  [[ " $PROBE_DATASETS " == *" $ds "* ]] && { cpus=8; probe="--probe --jobs 8"; }
  z=$(sbatch --parsable --account=$ACCOUNT --job-name=multiscale-step0-$ds --dependency=afterok:$e \
      --cpus-per-task=$cpus --mem=$mem --time=12:00:00 --output=logs/multiscale-step0-$ds-%j.log \
      --wrap="$PY optimization/multiscale_step0.py $common --step-ap $A/11-11-26 --seeds 0-2 $probe")
  echo "multiscale_extract0 $ds $e" >> "$OUT/job_ids_step0.txt"; echo "multiscale_step0 $ds $z" >> "$OUT/job_ids_step0.txt"
  echo "$ds: extract=$e step0=$z"
done
