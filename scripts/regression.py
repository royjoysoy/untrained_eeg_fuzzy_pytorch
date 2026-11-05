"""Regression: predict a continuous score (MMSE in ds004504) from every instance's frozen features.

The config's `regression` tasks: a participants.tsv column, optionally restricted to some rows
(`where`). One sample per participant (mean of its segments, as in the classification tasks).

Probe: standardize + ridge regression, 5-fold CV with the config's fixed split (KFold, shuffled).
Unlike the classifier's fixed C = 1, the ridge penalty is chosen in each training fold
(RidgeCV's leave-one-out over ALPHAS): R² depends on the scale of the predictions, which an
unsuited penalty ruins. Scores of the out-of-fold predictions: R² (main), MAE, Spearman's rho.

Every instance's probe gives, per extractor, three statistics as in permutation.py --ensemble:
  ensemble  R² of the instances' averaged predictions
  median    median R² over instances
  best      best R² over instances
With --permutations N, each is tested against the same statistic after shuffling the score
over participants and refitting every probe (the same shuffle for every extractor at
permutation i), and corrected over a task's extractors with max-T (Westfall-Young step-down
min-P), per task and statistic. Extractors in foundation.EXCLUDED (BENDR) are left out.

Writes results/phase1/<dataset>/regression_metrics.csv (one row per instance and task), and with
--permutations results/stats/regression_<dataset>.csv and its nulls in
results/stats/nulls_regression_<dataset>/ (one file per test, reused if the job is resubmitted, but
only if their fingerprint matches: features, scores, folds, penalties and permutation count).
    python scripts/regression.py --config configs/ds004504.yaml        (instances only, minutes)
    CONFIG=configs/ds004504.yaml ARGS="--permutations 10000" sbatch slurm/regression.sh
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.data import _data_dir, load_config
from untrained_eeg.foundation import EXCLUDED
from untrained_eeg.multitest import maxt

ALPHAS = np.logspace(-2, 6, 17)  # ridge penalties tried in each training fold
STATS = ("ensemble", "median", "best")


def task_data(f, column):
    """(X, y): each participant's mean features and score, for those with a score."""
    pid = f["participant_id"]
    people = [p for p in column.index if (pid == p).any()]
    return np.stack([f["feats"][pid == p].mean(axis=0) for p in people]), column[people].to_numpy(float)


def predict(X, y, cfg):
    """Out-of-fold ridge predictions with the config's fixed split."""
    cv = KFold(cfg["cv_folds"], shuffle=True, random_state=cfg["cv_seed"])
    return cross_val_predict(make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)), X, y, cv=cv)


def instance_stats(Xs, y, cfg):
    """Ensemble, median and best R² of one probe per instance."""
    preds = [predict(X, y, cfg) for X in Xs]
    r2 = [r2_score(y, p) for p in preds]
    return r2_score(y, np.mean(preds, axis=0)), np.median(r2), np.max(r2)


def null_key(Xs, y, cfg, n):
    """Fingerprint of everything a null depends on, so changed features or settings are not
    silently answered with old nulls."""
    h = hashlib.sha256()
    for a in (*Xs, y, ALPHAS, [cfg["cv_folds"], cfg["cv_seed"], n]):
        h.update(np.ascontiguousarray(a, dtype=float).tobytes())
    return h.hexdigest()


def null_stats(Xs, y, cfg, seeds):
    return [instance_stats(Xs, np.random.default_rng(s).permutation(y), cfg) for s in seeds]


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default=None)
parser.add_argument("--permutations", type=int, default=0, help="0: score the instances only")
parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
args = parser.parse_args()

cfg = load_config(args.config) if args.config else load_config()
ds = cfg["openneuro"]["id"]
root = Path(cfg["results_dir"]) / "phase1" / ds
participants = pd.read_csv(_data_dir(cfg) / "participants.tsv", sep="\t").set_index("participant_id")
targets = {}  # task: score per participant
for task, t in cfg["regression"].items():
    keep = np.ones(len(participants), bool)
    for col, values in t.get("where", {}).items():
        keep &= participants[col].isin(values).to_numpy()
    targets[task] = participants.loc[keep, t["column"]].dropna()

ext_dirs = sorted(p for p in (root / "features").iterdir() if p.is_dir() and p.name not in EXCLUDED)
metrics, observed = [], {}
for ext_dir in ext_dirs:
    files = [np.load(p) for p in sorted(ext_dir.glob("*.npz"))]
    for task, column in targets.items():
        Xs, y = zip(*(task_data(f, column) for f in files))
        y, preds = y[0], [predict(X, y[0], cfg) for X in Xs]
        for f, path, pred in zip(files, sorted(ext_dir.glob("*.npz")), preds):
            metrics.append(dict(extractor=ext_dir.name, group=str(f["group"]), instance=path.stem, task=task,
                                n=len(y), r2=r2_score(y, pred), mae=mean_absolute_error(y, pred),
                                spearman=spearmanr(y, pred).statistic))
        r2 = [r2_score(y, p) for p in preds]
        observed[ext_dir.name, task] = (r2_score(y, np.mean(preds, axis=0)), np.median(r2), np.max(r2))
        print(f"{ext_dir.name:16s} {task:14s} n {len(y)}  R² median {np.median(r2):6.3f}  best {np.max(r2):6.3f}  "
              f"ensemble {observed[ext_dir.name, task][0]:6.3f}", flush=True)
pd.DataFrame(metrics).to_csv(root / "regression_metrics.csv", index=False, float_format="%.6g")
print(f"wrote {root / 'regression_metrics.csv'}")
if not args.permutations:
    sys.exit()

nulls = Path(cfg["results_dir"]) / "stats" / f"nulls_regression_{ds}"
nulls.mkdir(parents=True, exist_ok=True)
rows = []
for ext_dir in ext_dirs:
    files = [np.load(p) for p in sorted(ext_dir.glob("*.npz"))]
    for task, column in targets.items():
        path = nulls / f"{ext_dir.name}__{task}.npz"
        Xs, y = zip(*(task_data(f, column) for f in files))
        key = null_key(Xs, y[0], cfg, args.permutations)
        if not path.exists() or str(np.load(path)["key"]) != key:
            chunk = max(1, min(50, args.permutations // (10 * args.workers)))
            parts = Parallel(n_jobs=args.workers)(
                delayed(null_stats)(Xs, y[0], cfg, range(s, min(s + chunk, args.permutations)))
                for s in range(0, args.permutations, chunk))
            np.savez(path, null=np.concatenate([np.array(p) for p in parts]), key=key)
        null = np.load(path)["null"]
        for i, stat in enumerate(STATS):
            obs = observed[ext_dir.name, task][i]
            rows.append(dict(dataset=ds, task=task, extractor=ext_dir.name, statistic=stat, instances=len(files),
                             r2=obs, null_mean=null[:, i].mean(), null_p95=np.quantile(null[:, i], 0.95),
                             n_permutations=len(null), p=(1 + (null[:, i] >= obs).sum()) / (1 + len(null))))
        print(f"{ext_dir.name:16s} {task:14s} median R² {observed[ext_dir.name, task][1]:6.3f}  "
              f"p {rows[-2]['p']:.4f}", flush=True)

out = pd.DataFrame(rows)
for (stat, task), g in out.groupby(["statistic", "task"]):  # joint nulls: same shuffles at each permutation
    col = STATS.index(stat)
    null = np.column_stack([np.load(nulls / f"{e}__{task}.npz")["null"][:, col] for e in g.extractor])
    out.loc[g.index, "p_maxt"] = maxt(null, g.r2.to_numpy())
out["significant"] = out.p_maxt < 0.05
dest = Path(cfg["results_dir"]) / "stats" / f"regression_{ds}.csv"
out.to_csv(dest, index=False, float_format="%.6g")
for stat, g in out.groupby("statistic"):
    print(f"wrote {dest}: {stat}: {g.significant.sum()} of {len(g)} significant after max-T (per task)")
