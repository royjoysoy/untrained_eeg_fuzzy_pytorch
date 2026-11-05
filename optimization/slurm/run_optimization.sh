#!/bin/bash
# Steps 1-3 on Rorqual. Submit ONLY after Roy approved the benchmark estimate.
#
#   export S=/project/rrg-glatard/shared/untrained_eeg
#   export OUT=$PWD/results/optimization/11-5-26          # your clone, not the shared folder
#   export VENV=$HOME/venv/uneeg_opt                        # see optimization/README.md
#   export DATASETS="ds003478 ds002778 ds003490 ds004504 lemon sleep_edf bnci2014_001"
#   bash optimization/slurm/run_optimization.sh
#
# Per dataset: 1 handcrafted job (8 CPUs) + an array over grid configs (1 CPU, all seeds),
# then 1 evaluation job (nested CV, 16 CPUs). Finally summarize + figures.
set -euo pipefail
: "${S:?}" "${OUT:?}" "${VENV:?}"
ACCOUNT=${ACCOUNT:-rrg-glatard_cpu}
SEEDS=${SEEDS:-0-29}
DATASETS=${DATASETS:-"ds003478 ds002778 ds003490 ds004504 lemon sleep_edf bnci2014_001"}
PY="source $VENV/bin/activate && python"
mkdir -p logs "$OUT"
N_CFG=$(bash -c "$PY -c 'import sys;sys.path.insert(0,\".\");from optimization.arch import ablation_grid;print(len(ablation_grid()))'")
evals=()
for ds in $DATASETS; do
  mem=16G; [ "$ds" = lemon ] && mem=40G
  common="--dataset $ds --data-root $S/data --out $OUT"
  h=$(sbatch --parsable --account=$ACCOUNT --job-name=opt-hand-$ds --cpus-per-task=8 --mem=$mem \
      --time=03:00:00 --output=logs/opt-hand-$ds-%j.log \
      --wrap="$PY optimization/extract.py handcrafted $common --jobs 8")
  c=$(sbatch --parsable --account=$ACCOUNT --job-name=opt-cnn-$ds --array=0-$((N_CFG-1)) \
      --cpus-per-task=1 --mem=$mem --time=06:00:00 --output=logs/opt-cnn-$ds-%A_%a.log \
      --wrap="$PY optimization/extract.py cnn $common --grid-index \$SLURM_ARRAY_TASK_ID --seeds $SEEDS")
  e=$(sbatch --parsable --account=$ACCOUNT --job-name=opt-eval-$ds --dependency=afterok:$h:$c \
      --cpus-per-task=16 --mem=32G --time=12:00:00 --output=logs/opt-eval-$ds-%j.log \
      --wrap="$PY optimization/evaluate_dataset.py $common --nested --jobs 16")
  evals+=("$e"); echo "$ds: handcrafted=$h cnn=$c eval=$e"
done
dep=$(IFS=:; echo "${evals[*]}")
sbatch --account=$ACCOUNT --job-name=opt-summary --dependency=afterok:$dep --cpus-per-task=1 --mem=8G \
  --time=00:30:00 --output=logs/opt-summary-%j.log \
  --wrap="$PY optimization/summarize.py --out $OUT --phase1 $S/results/phase1 && $PY optimization/report.py --out $OUT"
