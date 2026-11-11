#!/bin/bash
# Step (b): every preprocessing level (optimization/prep.py) for the two fixed architectures,
# the handcrafted features and MiniRocket; then nested and leave-one-dataset-out selection.
# Run on the LOGIN node, ONLY after Roy approved the estimate, and after
#   1. the raw-based caches were built (prep.py build, see optimization/README.md),
#   2. step 0 (prep_step0.py) printed MATCH on every dataset.
#
#   export S=/project/rrg-glatard/shared/untrained_eeg
#   export C=/project/rrg-glatard/royjoy/uneeg_prep_cache        # raw-based caches (bp05_45)
#   export OUT=$PWD/results/optimization/11-9-26                  # symlinked to /project
#   export VENV=$HOME/venv/uneeg_opt VENV_MR=$HOME/venv/uneeg_mr
#   bash optimization/slurm/run_prep.sh
#
# Per dataset, SLURM arrays: CNN (level x architecture x 10-seed chunk, 1 CPU), handcrafted
# (level, 8 CPUs), MiniRocket (level, 8 CPUs, 10 seeds), probes per level (16 CPUs), then one
# nested-selection job; at the end the summary and the compute budget.
set -euo pipefail
: "${S:?}" "${C:?}" "${OUT:?}" "${VENV:?}" "${VENV_MR:?}"
ACCOUNT=${ACCOUNT:-rrg-glatard_cpu}
DATASETS=${DATASETS:-"ds003478 ds002778 ds003490 ds004504 lemon sleep_edf bnci2014_001"}
ARCHS=(default "cov_channels=32,n_filters=256,readout=meanstd_ppv_max")  # phase-1 cnn1d, LODO pick of step (a)
CHUNKS=(0-9 10-19 20-29)
PY="export PYTHONUNBUFFERED=1 && source $VENV/bin/activate && python"
PY_MR="export PYTHONUNBUFFERED=1 && source $VENV_MR/bin/activate && python"
# the main environment must match phase 1 (numpy 2.4.2, scipy 1.17.1, scikit-learn 1.8.0, torch 2.14.1)
bash -c "$PY -c 'import numpy,scipy,sklearn,torch;v=(numpy.__version__,scipy.__version__,sklearn.__version__,torch.__version__);print(\"main env:\",v);assert v[0].startswith(\"2.4.2\") and v[1].startswith(\"1.17.1\") and v[2].startswith(\"1.8.0\") and v[3].startswith(\"2.14.1\"),\"versions differ from phase 1\"'"
mkdir -p logs "$OUT"
[ -e "$OUT/job_ids.txt" ] && { echo "$OUT/job_ids.txt exists: use a new OUT"; exit 1; }
log() { echo "$1 $2 $3" >> "$OUT/job_ids.txt"; }
lv_list() { bash -c "$PY optimization/prep.py levels --dataset $1"; }
pyfilter() { bash -c "$PY -c 'import sys;sys.path.insert(0,\".\");from optimization import prep;print(\" \".join(l for l in sys.argv[1:] if prep.$1(l)))' $2"; }
waits=()
for ds in $DATASETS; do
  mem=32G; [ "$ds" = lemon ] && mem=80G   # 256 Hz caches are twice the size
  read -r -a LV <<< "$(lv_list $ds)"
  read -r -a LV_H <<< "$(pyfilter for_handcrafted "${LV[*]}")"
  read -r -a LV_M <<< "$(pyfilter for_minirocket "${LV[*]}")"
  common="--dataset $ds --data-root $S/data --cache-root $C --out $OUT"
  nA=${#ARCHS[@]}; nC=${#CHUNKS[@]}; N=$(( ${#LV[@]} * nA * nC ))
  c=$(sbatch --parsable --account=$ACCOUNT --job-name=prep-cnn-$ds --array=0-$((N-1)) --cpus-per-task=1 \
      --mem=$mem --time=08:00:00 --output=logs/prep-cnn-$ds-%A_%a.log \
      --wrap="i=\$SLURM_ARRAY_TASK_ID; lv=(${LV[*]}); ar=(${ARCHS[*]}); ch=(${CHUNKS[*]}); \
              $PY optimization/extract.py cnn $common --prep \${lv[\$((i / ($nA * $nC)))]} \
              --configs \${ar[\$((i / $nC % $nA))]} --seeds \${ch[\$((i % $nC))]}")
  h=$(sbatch --parsable --account=$ACCOUNT --job-name=prep-hand-$ds --array=0-$((${#LV_H[@]}-1)) \
      --cpus-per-task=8 --mem=$mem --time=06:00:00 --output=logs/prep-hand-$ds-%A_%a.log \
      --wrap="lv=(${LV_H[*]}); $PY optimization/extract.py handcrafted $common --prep \${lv[\$SLURM_ARRAY_TASK_ID]} --jobs 8")
  m=$(sbatch --parsable --account=$ACCOUNT --job-name=prep-mr-$ds --array=0-$((${#LV_M[@]}-1)) \
      --cpus-per-task=8 --mem=$mem --time=24:00:00 --output=logs/prep-mr-$ds-%A_%a.log \
      --wrap="lv=(${LV_M[*]}); $PY_MR optimization/minirocket.py $common --prep \${lv[\$SLURM_ARRAY_TASK_ID]} --seeds 0-9 --jobs 8")
  e=$(sbatch --parsable --account=$ACCOUNT --job-name=prep-eval-$ds --array=0-$((${#LV[@]}-1)) \
      --dependency=afterok:$c:$h --cpus-per-task=16 --mem=32G --time=04:00:00 \
      --output=logs/prep-eval-$ds-%A_%a.log \
      --wrap="lv=(${LV[*]}); $PY optimization/evaluate_dataset.py --dataset $ds --data-root $S/data \
              --out $OUT/prep/\${lv[\$SLURM_ARRAY_TASK_ID]} --jobs 16")
  n=$(sbatch --parsable --account=$ACCOUNT --job-name=prep-nest-$ds --dependency=afterok:$e \
      --cpus-per-task=16 --mem=48G --time=12:00:00 --output=logs/prep-nest-$ds-%j.log \
      --wrap="$PY optimization/prep_select.py nested --dataset $ds --data-root $S/data --out $OUT --jobs 16")
  log prep_cnn $ds $c; log prep_handcrafted $ds $h; log prep_minirocket $ds $m
  log prep_evaluate $ds $e; log prep_nested $ds $n
  waits+=("$n" "$m"); echo "$ds: levels=${#LV[@]} cnn=$c hand=$h mr=$m eval=$e nested=$n"
done
dep=$(IFS=:; echo "${waits[*]}")
sm=$(sbatch --parsable --account=$ACCOUNT --job-name=prep-summary --dependency=afterok:$dep --mem=8G \
  --time=00:30:00 --output=logs/prep-summary-%j.log \
  --wrap="$PY optimization/prep_select.py summary --out $OUT")
log prep_summary all $sm
all=$(cut -d' ' -f3 "$OUT/job_ids.txt" | paste -sd:)
sbatch --account=$ACCOUNT --job-name=prep-cost --dependency=afterany:$all --cpus-per-task=1 --mem=2G \
  --time=00:15:00 --output=logs/prep-cost-%j.log --wrap="$PY optimization/compute_cost.py --out $OUT"
echo "summary=$sm  (results in $OUT/prep_summary/headline.csv)"
