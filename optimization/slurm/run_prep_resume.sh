#!/bin/bash
# Step (b): re-run the probes of some levels of ONE dataset (e.g. after a TIMEOUT), then that
# dataset's nested selection and the summary. Run on the LOGIN node:
#
#   bash optimization/slurm/run_prep_resume.sh ds003478 rej_mad5 bp05_45
#
# Levels whose per_seed.csv already exists are refused by evaluate_dataset.py (never overwrites).
set -euo pipefail
ds=${1:?dataset}; shift
[ $# -gt 0 ] || { echo "give the levels to re-run"; exit 1; }
S=${S:-/project/rrg-glatard/shared/untrained_eeg}
OUT=${OUT:-$PWD/results/optimization/11-9-26}
ACCOUNT=${ACCOUNT:-rrg-glatard_cpu}
PY="export PYTHONUNBUFFERED=1 && source ${VENV:-$HOME/venv/uneeg_opt}/bin/activate && python"
mkdir -p logs
ids=()
for lv in "$@"; do
  e=$(sbatch --parsable --account=$ACCOUNT --job-name=prep-eval2-$ds --cpus-per-task=16 --mem=32G \
      --time=12:00:00 --output=logs/prep-eval2-$ds-$lv-%j.log \
      --wrap="$PY optimization/evaluate_dataset.py --dataset $ds --data-root $S/data --out $OUT/prep/$lv --jobs 16")
  echo "prep_evaluate $ds $e" >> "$OUT/job_ids.txt"; ids+=("$e"); echo "eval $lv: $e"
done
dep=$(IFS=:; echo "${ids[*]}")
n=$(sbatch --parsable --account=$ACCOUNT --job-name=prep-nest-$ds --dependency=afterok:$dep --cpus-per-task=16 \
    --mem=48G --time=12:00:00 --output=logs/prep-nest-$ds-%j.log \
    --wrap="$PY optimization/prep_select.py nested --dataset $ds --data-root $S/data --out $OUT --jobs 16")
sm=$(sbatch --parsable --account=$ACCOUNT --job-name=prep-summary --dependency=afterok:$n --mem=8G \
     --time=00:30:00 --output=logs/prep-summary-%j.log \
     --wrap="$PY optimization/prep_select.py summary --out $OUT")
echo "prep_nested $ds $n" >> "$OUT/job_ids.txt"; echo "prep_summary all $sm" >> "$OUT/job_ids.txt"
echo "nested: $n  summary: $sm"
