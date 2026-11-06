#!/bin/bash
# Step 2, last part: every combination of the strongest factors (optimization/search/*.txt),
# extracted for all seeds (3 seed chunks per config run in parallel), then nested and
# leave-one-dataset-out selection among them. Run on the LOGIN node after approval:
#
#   export S=/project/rrg-glatard/shared/untrained_eeg OUT=$PWD/results/optimization/11-5-26b
#   export VENV=$HOME/venv/uneeg_opt LIST=optimization/search/configs_11-6-26.txt
#   bash optimization/slurm/run_search.sh
set -euo pipefail
: "${S:?}" "${OUT:?}" "${VENV:?}" "${LIST:?}"
ACCOUNT=${ACCOUNT:-rrg-glatard_cpu}
DATASETS=${DATASETS:-"ds003478 ds002778 ds003490 ds004504 lemon sleep_edf bnci2014_001"}
PY="export PYTHONUNBUFFERED=1 && source $VENV/bin/activate && python"
CHUNKS=(0-9 10-19 20-29)
mapfile -t CFG < "$LIST"
N=$(( ${#CFG[@]} * ${#CHUNKS[@]} ))
cp "$LIST" "$OUT/search_configs.txt"
log() { echo "$1 $2 $3" >> "$OUT/job_ids.txt"; }
evals=()
for ds in $DATASETS; do
  mem=16G; [ "$ds" = lemon ] && mem=40G
  common="--dataset $ds --data-root $S/data --out $OUT"
  c=$(sbatch --parsable --account=$ACCOUNT --job-name=srch-cnn-$ds --array=0-$((N-1)) \
      --cpus-per-task=1 --mem=$mem --time=03:00:00 --output=logs/srch-cnn-$ds-%A_%a.log \
      --wrap="i=\$SLURM_ARRAY_TASK_ID; cfgs=(${CFG[*]}); chunks=(${CHUNKS[*]}); \
              $PY optimization/extract.py cnn $common --configs \${cfgs[\$((i / ${#CHUNKS[@]}))]} \
              --seeds \${chunks[\$((i % ${#CHUNKS[@]}))]}")
  e=$(sbatch --parsable --account=$ACCOUNT --job-name=srch-eval-$ds --dependency=afterok:$c \
      --cpus-per-task=16 --mem=32G --time=04:00:00 --output=logs/srch-eval-$ds-%j.log \
      --wrap="$PY optimization/evaluate_dataset.py $common --search-list $OUT/search_configs.txt --jobs 16")
  log search_cnn $ds $c; log search_evaluate $ds $e; evals+=("$e")
  echo "$ds: cnn=$c eval=$e"
done
dep=$(IFS=:; echo "${evals[*]}")
sm=$(sbatch --parsable --account=$ACCOUNT --job-name=srch-summary --dependency=afterok:$dep --mem=8G \
  --time=00:30:00 --output=logs/srch-summary-%j.log \
  --wrap="$PY optimization/summarize.py --out $OUT --phase1 $S/results/phase1 --dest summary_search \
          --search-list $OUT/search_configs.txt")
log search_summary all $sm
echo "summary=$sm  (results in $OUT/summary_search/headline.csv)"
