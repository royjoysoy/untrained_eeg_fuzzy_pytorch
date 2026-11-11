#!/bin/bash
# Step (a'): every readout level (optimization/readout.py) for the two fixed architectures,
# 30 weight seeds, the reference preprocessing; then nested and leave-one-dataset-out selection.
# Handcrafted features and MiniRocket are NOT rerun: the summary takes them from step (b)'s
# reference level (same data, samples and folds).
# Run on the LOGIN node, ONLY after Roy approved the estimate (readout_estimate.py) and after
# step 0 (run_readout_step0.sh) printed MATCH on every dataset:
#
#   export S=/project/rrg-glatard/shared/untrained_eeg
#   export A=/project/rrg-glatard/royjoy/uneeg_opt_results/optimization   # steps (a) and (b)
#   export OUT=$PWD/results/optimization/11-11-26                          # same OUT as step 0
#   export VENV=$HOME/venv/uneeg_opt
#   bash optimization/slurm/run_readout.sh
#
# Per dataset, SLURM arrays: CNN (level x architecture x 10-seed chunk, 1 CPU; step 0's files
# are skipped, never overwritten), probes per level (16 CPUs, 12 h), then one nested-selection
# job; at the end the summary and the compute budget.
set -euo pipefail
: "${S:?}" "${A:?}" "${OUT:?}" "${VENV:?}"
ACCOUNT=${ACCOUNT:-rrg-glatard_cpu}
DATASETS=${DATASETS:-"ds003478 ds002778 ds003490 ds004504 lemon sleep_edf bnci2014_001"}
ARCHS=(default "cov_channels=32,n_filters=256,readout=meanstd_ppv_max")  # phase-1 cnn1d, LODO pick of step (a)
CHUNKS=(0-9 10-19 20-29)
PY="export PYTHONUNBUFFERED=1 && source $VENV/bin/activate && python"
# the main environment must match phase 1 (numpy 2.4.2, scipy 1.17.1, scikit-learn 1.8.0, torch 2.14.1)
bash -c "$PY -c 'import numpy,scipy,sklearn,torch;v=(numpy.__version__,scipy.__version__,sklearn.__version__,torch.__version__);print(\"main env:\",v);assert v[0].startswith(\"2.4.2\") and v[1].startswith(\"1.17.1\") and v[2].startswith(\"1.8.0\") and v[3].startswith(\"2.14.1\"),\"versions differ from phase 1\"'"
[ -d "$A/11-9-26/prep/ref" ] || { echo "step (b) baselines not found: $A/11-9-26/prep/ref"; exit 1; }
mkdir -p logs "$OUT"
[ -e "$OUT/job_ids.txt" ] && { echo "$OUT/job_ids.txt exists: already submitted (check squeue)"; exit 1; }
log() { echo "$1 $2 $3" >> "$OUT/job_ids.txt"; }
waits=()
for ds in $DATASETS; do
  mem=16G; [ "$ds" = lemon ] && mem=80G
  read -r -a LV <<< "$(bash -c "$PY optimization/readout.py levels --dataset $ds")"
  common="--dataset $ds --data-root $S/data --out $OUT"
  nA=${#ARCHS[@]}; nC=${#CHUNKS[@]}; N=$(( ${#LV[@]} * nA * nC ))
  c=$(sbatch --parsable --account=$ACCOUNT --job-name=readout-cnn-$ds --array=0-$((N-1)) --cpus-per-task=1 \
      --mem=$mem --time=08:00:00 --output=logs/readout-cnn-$ds-%A_%a.log \
      --wrap="i=\$SLURM_ARRAY_TASK_ID; lv=(${LV[*]}); ar=(${ARCHS[*]}); ch=(${CHUNKS[*]}); \
              $PY optimization/extract.py cnn $common --readout \${lv[\$((i / ($nA * $nC)))]} \
              --configs \${ar[\$((i / $nC % $nA))]} --seeds \${ch[\$((i % $nC))]}")
  e=$(sbatch --parsable --account=$ACCOUNT --job-name=readout-eval-$ds --array=0-$((${#LV[@]}-1)) \
      --dependency=afterok:$c --cpus-per-task=16 --mem=32G --time=12:00:00 \
      --output=logs/readout-eval-$ds-%A_%a.log \
      --wrap="lv=(${LV[*]}); $PY optimization/evaluate_dataset.py --dataset $ds --data-root $S/data \
              --out $OUT/readout/\${lv[\$SLURM_ARRAY_TASK_ID]} --jobs 16")
  n=$(sbatch --parsable --account=$ACCOUNT --job-name=readout-nest-$ds --dependency=afterok:$c \
      --cpus-per-task=16 --mem=48G --time=24:00:00 --output=logs/readout-nest-$ds-%j.log \
      --wrap="$PY optimization/prep_select.py nested --study readout --dataset $ds --data-root $S/data --out $OUT --jobs 16")
  log readout_cnn $ds $c; log readout_evaluate $ds $e; log readout_nested $ds $n
  waits+=("$e" "$n"); echo "$ds: levels=${#LV[@]} cnn=$c eval=$e nested=$n"
done
dep=$(IFS=:; echo "${waits[*]}")
sm=$(sbatch --parsable --account=$ACCOUNT --job-name=readout-summary --dependency=afterok:$dep --mem=8G \
  --time=00:30:00 --output=logs/readout-summary-%j.log \
  --wrap="$PY optimization/prep_select.py summary --study readout --out $OUT --baselines $A/11-9-26")
log readout_summary all $sm
all=$(cut -d' ' -f3 "$OUT/job_ids.txt" | paste -sd:)
sbatch --account=$ACCOUNT --job-name=readout-cost --dependency=afterany:$all --cpus-per-task=1 --mem=2G \
  --time=00:15:00 --output=logs/readout-cost-%j.log --wrap="$PY optimization/compute_cost.py --out $OUT"
echo "summary=$sm  (results in $OUT/readout_summary/headline.csv)"
