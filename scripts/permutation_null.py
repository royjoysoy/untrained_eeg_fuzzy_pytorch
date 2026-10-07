"""Group C (#32, task 9): label-permutation null for the BDI probe AUC.

For one saved run (default w000_rn: weight seed 0, round to nearest) and each model and
condition, the observed out-of-fold AUC (fixed CV split, as in the pilot) is compared with
the AUCs of N random permutations of the BDI labels through the same probe and split.
p = (1 + #{permuted AUC >= observed}) / (1 + N).

Writes <out>/permutation_null.csv (one row per model x condition: observed AUC, null mean,
null 95th percentile, p) and <out>/null_aucs.csv.gz (every permuted AUC).

    python scripts/permutation_null.py --config configs/paper.yaml \\
        --features results/paper/gpu_features --out results/paper/C9 --n-perm 1000 --workers 16
"""

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.data import load_config, load_participants
from untrained_eeg.models import MODELS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_runs import participant_features, probe  # noqa: E402

_job = {}


def _one_thread():
    """One BLAS thread per worker: the pool is the parallelism."""
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def _null_auc(i):
    X, y, cv_seed, folds, perm_seed = _job["X"], _job["y"], _job["cv_seed"], _job["folds"], _job["seed"]
    yp = np.random.default_rng([perm_seed, i]).permutation(y)
    return roc_auc_score(yp, probe(X, yp, cv_seed, folds))


def permutation_test(X, y, cv_seed, folds, n_perm, workers, seed=0):
    """Observed AUC, all permuted AUCs and the permutation p-value."""
    observed = roc_auc_score(y, probe(X, y, cv_seed, folds))
    _job.update(X=X, y=y, cv_seed=cv_seed, folds=folds, seed=seed)  # inherited by forked workers
    with Pool(workers, initializer=_one_thread) as pool:
        null = np.array(pool.map(_null_auc, range(n_perm), chunksize=max(1, n_perm // (4 * workers))))
    return observed, null, (1 + (null >= observed).sum()) / (1 + n_perm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--features", required=True)
    parser.add_argument("--features-run", default="w000_rn")
    parser.add_argument("--out", required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=None)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    labels = load_participants(cfg).set_index("participant_id")["label"]
    rows, nulls = [], []
    for model in args.models or cfg["models"]:
        f = np.load(Path(args.features) / model / f"{args.features_run}.npz")
        for condition in cfg["conditions"]:
            feats = participant_features(f, condition)
            pids = [p for p in labels.index if p in feats]
            X, y = np.stack([feats[p] for p in pids]), labels[pids].to_numpy()
            obs, null, p = permutation_test(X, y, cfg["cv_seed"], cfg["cv_folds"], args.n_perm, args.workers)
            rows.append(dict(model=model, condition=condition, features_run=args.features_run, n=len(y),
                             auc=obs, null_mean=null.mean(), null_q95=np.quantile(null, 0.95), p=p,
                             n_perm=args.n_perm))
            nulls.append(pd.DataFrame(dict(model=model, condition=condition, perm=np.arange(len(null)), auc=null)))
            print(f"{model:8s} {condition:6s} AUC {obs:.3f}  null {null.mean():.3f} (95% {np.quantile(null, .95):.3f})"
                  f"  p = {p:.3f}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "permutation_null.csv", index=False, float_format="%.6g")
    pd.concat(nulls).to_csv(out / "null_aucs.csv.gz", index=False, float_format="%.6g")
    print(f"wrote {out}")
