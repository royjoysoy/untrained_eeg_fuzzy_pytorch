#!/bin/bash
set -euo pipefail
if command -v module >/dev/null 2>&1; then
    module load "${APPTAINER_MODULE:-apptainer/1.3.5}"
fi
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
image=${FUZZY_IMAGE:-"$base/fuzzy-v2.6.0-pytorch2.2.1-avx2.sif"}
image_hash=$(sha256sum "$image")
image_hash=${image_hash%% *}
# PRISM initializes random streams during library loading; the runtime setter
# does not reset existing streams. Seed the process before importing PyTorch.
prism_seed=1001
seed_next=false
for argument in "$@"; do
    if "$seed_next"; then
        prism_seed=$argument
        seed_next=false
    elif [[ "$argument" == --perturbation-seed ]]; then
        seed_next=true
    elif [[ "$argument" == --perturbation-seed=* ]]; then
        prism_seed=${argument#*=}
    fi
done
if "$seed_next" || [[ ! "$prism_seed" =~ ^[0-9]+$ ]]; then
    echo 'Supply an integer --perturbation-seed' >&2
    exit 2
fi
exec apptainer exec --cleanenv --containall --no-mount hostfs \
    --bind "$base:/work" --pwd /work \
    --env PYTHONNOUSERSITE=1 --env PYTHONHASHSEED=0 \
    --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 --env NUMEXPR_NUM_THREADS=1 \
    --env VFC_BACKENDS=libinterflop_ieee.so \
    --env PRISM_SEED="$prism_seed" \
    --env PYTHONPATH=/work/runtime-dependencies:/usr/local/lib/python3.12/dist-packages:/usr/local/lib/python3.12/site-packages \
    --env FUZZY_IMAGE_SHA256="$image_hash" \
    "$image" python3 "$@"
