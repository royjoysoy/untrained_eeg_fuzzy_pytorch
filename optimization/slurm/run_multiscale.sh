#!/bin/bash
# Step (a''): every multiscale / filter-bank level (optimization/multiscale.py) of combined + Hjorth,
# 30 weight seeds, the reference preprocessing; then nested and leave-one-dataset-out selection.
# Handcrafted features and MiniRocket are NOT rerun: the summary takes them from step (b)'s
# reference level (same data, samples and folds).
# Run on the LOGIN node, ONLY after Roy approved the estimate (multiscale_estimate.py) and after
# step 0 (run_multiscale_step0.sh) printed MATCH on every dataset:
#
#   export S=/project/rrg-glatard/shared/untrained_eeg
#   export A=/project/rrg-glatard/royjoy/uneeg_opt_results/optimization   # steps (a), (b), (a')
#   export OUT=$PWD/results/optimization/11-13-26                          # same OUT as step 0
#   export VENV=$HOME/venv/uneeg_opt
#   bash optimization/slurm/run_multiscale.sh
#
# Per dataset, SLURM arrays: CNN (level x 10-seed chunk, 1 CPU; step 0's files are skipped, never
# overwritten), probes per level (16 CPUs, 12 h), then one nested-selection job; at the end the
# summary and the compute budget.
set -euo pipefail
: "${S:?}" "${A:?}" "${OUT:?}" "${VENV:?}"
ACCOUNT=${ACCOUNT:-rrg-glatard_cpu}
DATASETS=${DATASETS:-"ds003478 ds002778 ds003490 ds004504 lemon sleep_edf bnci2014_001"}
BASE="n_filters=256,readout=hjorth"   # combined + Hjorth (optimization/multiscale.py BASE)
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
  mem=16G; nmem=48G; [ "$ds" = lemon ] && mem=80G; [ "$ds" = sleep_edf ] && nmem=64G
  read -r -a LV <<< "$(bash -c "$PY optimization/multiscale.py levels --dataset $ds")"
  common="--dataset $ds --data-root $S/data --out $OUT"
  nC=${#CHUNKS[@]}; N=$(( ${#LV[@]} * nC ))
  # 12 h: the filter bank makes the first convolution 6x wider (LEMON's longest chunk was 1.1 h at ref)
  c=$(sbatch --parsable --account=$ACCOUNT --job-name=multiscale-cnn-$ds --array=0-$((N-1)) --cpus-per-task=1 \
      --mem=$mem --time=12:00:00 --output=logs/multiscale-cnn-$ds-%A_%a.log \
      --wrap="i=\$SLURM_ARRAY_TASK_ID; lv=(${LV[*]}); ch=(${CHUNKS[*]}); \
              $PY optimization/extract.py cnn $common --multiscale \${lv[\$((i / $nC))]} \
              --configs $BASE --seeds \${ch[\$((i % $nC))]}")
  e=$(sbatch --parsable --account=$ACCOUNT --job-name=multiscale-eval-$ds --array=0-$((${#LV[@]}-1)) \
      --dependency=afterok:$c --cpus-per-task=16 --mem=32G --time=12:00:00 \
      --output=logs/multiscale-eval-$ds-%A_%a.log \
      --wrap="lv=(${LV[*]}); $PY optimization/evaluate_dataset.py --dataset $ds --data-root $S/data \
              --out $OUT/multiscale/\${lv[\$SLURM_ARRAY_TASK_ID]} --jobs 16")
  n=$(sbatch --parsable --account=$ACCOUNT --job-name=multiscale-nest-$ds --dependency=afterok:$c \
      --cpus-per-task=16 --mem=$nmem --time=24:00:00 --output=logs/multiscale-nest-$ds-%j.log \
      --wrap="$PY optimization/prep_select.py nested --study multiscale --dataset $ds --data-root $S/data --out $OUT --jobs 16")
  log multiscale_cnn $ds $c; log multiscale_evaluate $ds $e; log multiscale_nested $ds $n
  waits+=("$e" "$n"); echo "$ds: levels=${#LV[@]} cnn=$c eval=$e nested=$n"
done
dep=$(IFS=:; echo "${waits[*]}")
sm=$(sbatch --parsable --account=$ACCOUNT --job-name=multiscale-summary --dependency=afterok:$dep --mem=8G \
  --time=00:30:00 --output=logs/multiscale-summary-%j.log \
  --wrap="$PY optimization/prep_select.py summary --study multiscale --out $OUT --baselines $A/11-9-26")
log multiscale_summary all $sm
all=$(cut -d' ' -f3 "$OUT/job_ids.txt" | paste -sd:)
sbatch --account=$ACCOUNT --job-name=multiscale-cost --dependency=afterany:$all --cpus-per-task=1 --mem=2G \
  --time=00:15:00 --output=logs/multiscale-cost-%j.log --wrap="$PY optimization/compute_cost.py --out $OUT"
echo "summary=$sm  (results in $OUT/multiscale_summary/headline.csv)"
