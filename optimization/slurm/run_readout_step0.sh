#!/bin/bash
# Step (a'), before the main run: per dataset, extract
#   - the reference readout (seeds 0-2) of both architectures -> must equal step (a) bit for bit,
#   - every other readout level (seed 0) -> sanity checks on real data (readout_step0.py),
# then check, and on PROBE_DATASETS also time the probe of seed 0 at every level (for the
# CPU-hour estimate). Run on the LOGIN node:
#
#   export S=/project/rrg-glatard/shared/untrained_eeg
#   export A=/project/rrg-glatard/royjoy/uneeg_opt_results/optimization   # step (a) results
#   export OUT=$PWD/results/optimization/11-11-26
#   export VENV=$HOME/venv/uneeg_opt
#   bash optimization/slurm/run_readout_step0.sh
# Then: grep "STEP 0" logs/readout-step0-*.log   (every dataset must say MATCH)
set -euo pipefail
: "${S:?}" "${A:?}" "${OUT:?}" "${VENV:?}"
ACCOUNT=${ACCOUNT:-rrg-glatard_cpu}
DATASETS=${DATASETS:-"ds003478 ds002778 ds003490 ds004504 lemon sleep_edf bnci2014_001"}
PROBE_DATASETS=${PROBE_DATASETS:-"ds004504 bnci2014_001"}   # small ones: timing of every level
COMBINED="cov_channels=32,n_filters=256,readout=meanstd_ppv_max"
LEVELS="logvar logpow_first ref_logvar subwin across_win hjorth tcov32 ref_tcov32 tcov16"
PY="export PYTHONUNBUFFERED=1 && source $VENV/bin/activate && python"
# the main environment must match phase 1 (numpy 2.4.2, scipy 1.17.1, scikit-learn 1.8.0, torch 2.14.1)
bash -c "$PY -c 'import numpy,scipy,sklearn,torch;v=(numpy.__version__,scipy.__version__,sklearn.__version__,torch.__version__);print(\"main env:\",v);assert v[0].startswith(\"2.4.2\") and v[1].startswith(\"1.17.1\") and v[2].startswith(\"1.8.0\") and v[3].startswith(\"2.14.1\"),\"versions differ from phase 1\"'"
mkdir -p logs "$OUT"
[ -e "$OUT/job_ids_step0.txt" ] && { echo "$OUT/job_ids_step0.txt exists: already submitted (check squeue)"; exit 1; }
for ds in $DATASETS; do
  mem=16G; [ "$ds" = lemon ] && mem=80G
  common="--dataset $ds --data-root $S/data --out $OUT"
  # 1 CPU: extraction is single-threaded (as in steps a and b)
  e=$(sbatch --parsable --account=$ACCOUNT --job-name=readout-ext0-$ds --cpus-per-task=1 --mem=$mem \
      --time=12:00:00 --output=logs/readout-ext0-$ds-%j.log \
      --wrap="$PY optimization/extract.py cnn $common --readout ref --configs default $COMBINED --seeds 0-2 && \
              for lv in $LEVELS; do $PY optimization/extract.py cnn $common --readout \$lv --configs default $COMBINED --seeds 0 || exit 1; done")
  cpus=1; probe=""
  [[ " $PROBE_DATASETS " == *" $ds "* ]] && { cpus=8; probe="--probe --jobs 8"; }
  z=$(sbatch --parsable --account=$ACCOUNT --job-name=readout-step0-$ds --dependency=afterok:$e \
      --cpus-per-task=$cpus --mem=$mem --time=12:00:00 --output=logs/readout-step0-$ds-%j.log \
      --wrap="$PY optimization/readout_step0.py $common --step-a $A/11-5-26 $A/11-5-26b --seeds 0-2 $probe")
  echo "readout_extract0 $ds $e" >> "$OUT/job_ids_step0.txt"; echo "readout_step0 $ds $z" >> "$OUT/job_ids_step0.txt"
  echo "$ds: extract=$e step0=$z"
done
