#!/bin/bash
# Run a script with a native (non-container) Fuzzy PyTorch build, e.g. the AArch64 one.
#
#   export FUZZY_ENV=/path/to/fuzzy-pytorch-arm/env.sh   # sets PATH / libs for the build
#   MODE=rn bash environment/fuzzy_arm.sh scripts/02_seed_variability.py --config ...
#   MODE=sr SEED=7 bash environment/fuzzy_arm.sh scripts/03_mca_variability.py --sample 7
#
#   MODE     rn (round to nearest, the reference) or sr (stochastic rounding). Default rn.
#   SEED     random seed of the stochastic rounding (one per sample). Default 1.
#   THREADS  torch / BLAS threads. Default 1; keep it fixed across runs you compare.
#   FUZZY_PKGS  extra packages from environment/requirements-fuzzy.txt. Default $HOME/fuzzy_pkgs.
set -euo pipefail
: "${FUZZY_ENV:?export FUZZY_ENV=/path/to/the/build/env.sh}"
source "$FUZZY_ENV"
export PYTHONPATH="${FUZZY_PKGS:-$HOME/fuzzy_pkgs}"
# Same PRISM backend in both modes, so rn and sr runs share one code path.
export VFC_BACKENDS="libinterflop_prism.so --seed=${SEED:-1} --mode=${MODE:-rn}"
export OMP_NUM_THREADS=${THREADS:-1} OPENBLAS_NUM_THREADS=${THREADS:-1} MKL_NUM_THREADS=${THREADS:-1}
exec python -u "$@"
