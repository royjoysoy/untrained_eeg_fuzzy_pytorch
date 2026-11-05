"""Phase 2: ensembles of K model instances on one dataset, and how they change with K.

A pool is a set of instances: the 30 seeds of one untrained extractor, the pretrained
checkpoints, or a mix of extractors. For each K (1 .. pool size) and up to --draws distinct
sets of K instances (all of them when fewer exist, fixed RNG), the K instances are combined:
  features   average their block features, then one probe (same extractor only)
  average    average their out-of-fold P(class 1) (soft voting)
  vote       majority vote of their predicted classes; the score is the share voting
             class 1, ties (even K) go to the average P
All instances share the same fixed CV split (cv_seed of the config), so their out-of-fold
predictions can be combined sample by sample.

Reads results/phase1/<dataset>/ (features/ and predictions.csv.gz). Writes results/phase2/<dataset>/:
  draws.csv     pool, method, task, k, draw, instances, auc
  summary.csv   per pool x method x task x k: AUC mean and SD over draws, median per-sample
                SD of the score over draws, disagreement (chance that two random draws predict
                different classes for a sample, averaged over samples; it does not depend on
                the number of draws, which changes with K; undefined for a single draw)
  curves.png    AUC and spread against K

    python scripts/phase2_ensembles.py --workers 8                   # ds003478
    python scripts/phase2_ensembles.py --config configs/ds004504.yaml --workers 8
"""

import argparse
import itertools
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.data import load_config
from untrained_eeg.foundation import EXCLUDED, FOUNDATION
from untrained_eeg.tasks import MULTICLASS, auc_score, predicted, probe, task_data, task_labels, task_names

UNTRAINED = ["cnn1d", "shallow", "eegnet", "eegnet_3ch", "shallow_bcic2a", "scattering"]
PRETRAINED = ["eegnetv4", "shallow_pretrained"]
FMS = [e for e in FOUNDATION if e not in EXCLUDED]  # BENDR's near-constant features are left out
POOLS = {**{e: [e] for e in UNTRAINED + PRETRAINED},  # pool name: extractors in it
         "untrained_mix": ["cnn1d", "shallow", "eegnet"],  # different architectures, all channels
         "pretrained_mix": PRETRAINED,
         "foundation_mix": FMS,  # one instance per foundation model
         "all_pretrained_mix": PRETRAINED + FMS}


def draws_of(n, k, n_draws, rng):
    """Up to n_draws distinct sets of k indices out of n: all of them when there are few."""
    if math.comb(n, k) <= n_draws:
        return [list(c) for c in itertools.combinations(range(n), k)]
    out = set()
    while len(out) < n_draws:
        out.add(tuple(sorted(rng.choice(n, size=k, replace=False))))
    return [list(d) for d in sorted(out)]


def vote(probs):
    """Majority vote of the instances' classes (axis 0).

    Two classes: score = share voting 1, ties go to the mean P. Several classes (probs
    instances x samples x classes): score = share voting each class, ties to the mean P.
    """
    if probs.ndim == 3:
        votes = np.eye(probs.shape[2])[probs.argmax(axis=2)].mean(axis=0)
        return votes + 1e-9 * probs.mean(axis=0)
    share = (probs > 0.5).mean(axis=0)
    tie = share == 0.5
    share[tie] = np.where(probs[:, tie].mean(axis=0) > 0.5, 0.5 + 1e-9, 0.5)
    return share


def disagreement(cls):
    """Chance that two different draws predict different classes, averaged over samples.

    cls: draws x samples predicted classes. With p_c the share of the n draws predicting class c,
    the unbiased chance for a sample is (1 - sum_c p_c^2) * n / (n - 1); 2 p (1 - p) * n / (n - 1)
    for two classes.
    """
    n = len(cls)
    shares = np.stack([(cls == c).mean(axis=0) for c in np.unique(cls)])
    return ((1 - (shares**2).sum(axis=0)) * n / (n - 1)).mean()


def feature_ensemble(X, y, groups, task, cfg):
    """Probe one draw's averaged features (computed by the caller: workers get only X)."""
    return probe(X, y, groups, task, cfg["cv_seed"], cfg["cv_folds"])[0]


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default=None)
parser.add_argument("--draws", type=int, default=50, help="draws per K for average and vote")
parser.add_argument("--feature-draws", type=int, default=20, help="draws per K for features (each a probe)")
parser.add_argument("--max-k", type=int, default=30)
parser.add_argument("--workers", type=int, default=1)
args = parser.parse_args()

cfg = load_config(args.config) if args.config else load_config()
src = Path(cfg["results_dir"]) / "phase1" / cfg["openneuro"]["id"]
out = Path(cfg["results_dir"]) / "phase2" / cfg["openneuro"]["id"]
out.mkdir(parents=True, exist_ok=True)
labels = task_labels(cfg)
TASKS = task_names(cfg)
preds = pd.read_csv(src / "predictions.csv.gz")
preds = preds[preds.cv_seed == cfg["cv_seed"]]
rng = np.random.default_rng(0)
rows, summary = [], []

for pool, extractors in POOLS.items():
    extractors = [e for e in extractors if e in set(preds.extractor)]  # those this dataset has
    p = preds[preds.extractor.isin(extractors)]
    if p[["extractor", "instance"]].drop_duplicates().shape[0] < 2:  # nothing to combine
        continue
    insts = sorted(set(zip(p.extractor, p.instance)))  # (extractor, instance) members
    feats = None
    if len(extractors) == 1:  # features can be averaged only within one extractor
        files = [src / "features" / e / f"{i}.npz" for e, i in insts]
        f0 = {k: np.load(files[0])[k] for k in ("cond", "participant_id")}  # arrays: workers get copies
        feats = [np.load(f)["feats"] for f in files]
        if len({x.shape for x in feats}) > 1:  # e.g. scattering: feature count depends on J
            feats = None
    for task in TASKS:
        t = p[p.task == task]
        if task in MULTICLASS:  # instances x samples x classes
            n_cls = len(MULTICLASS[task])
            wides = [t.pivot_table(index="sample", columns=["extractor", "instance"], values=f"prob_{c}")[insts]
                     for c in range(n_cls)]
            wide = wides[0]
            P = np.stack([w.to_numpy().T for w in wides], axis=2)
        else:  # instances x samples
            wide = t.pivot_table(index="sample", columns=["extractor", "instance"], values="prob")[insts]
            P = wide.to_numpy().T
        if wide.isna().any().any():  # e.g. extractors whose preprocessing kept different blocks
            raise ValueError(f"{pool} {task}: the instances do not have the same samples")
        y = t.drop_duplicates("sample").set_index("sample").label.loc[wide.index].to_numpy()
        features_ok = feats is not None and cfg.get("ensemble_features", True)  # off where probes are slow
        methods = ["average", "vote"] + (["features"] if features_ok else [])
        for k in range(1, min(len(insts), args.max_k) + 1):
            for method in methods:
                draws = draws_of(len(insts), k, args.feature_draws if method == "features" else args.draws, rng)
                if method == "features":
                    inputs = [task_data(np.mean([feats[m] for m in d], axis=0), f0["cond"],
                                        f0["participant_id"], task, labels) for d in draws]
                    scores = Parallel(n_jobs=args.workers)(
                        delayed(feature_ensemble)(X, yy, g, task, cfg) for X, yy, g, _ in inputs)
                    order = pd.Index(inputs[0][3]).get_indexer(wide.index)  # samples in the order of P
                    assert (order >= 0).all(), "feature samples differ from the predictions'"
                    scores = np.array([s[order] for s in scores])
                elif method == "average":
                    scores = np.array([P[d].mean(axis=0) for d in draws])
                else:
                    scores = np.array([vote(P[d]) for d in draws])
                aucs = np.array([auc_score(y, s) for s in scores])
                cls = np.array([predicted(s) for s in scores])
                for i, (d, a) in enumerate(zip(draws, aucs)):
                    rows.append(dict(pool=pool, method=method, task=task, k=k, draw=i, auc=a,
                                     instances=" ".join(f"{insts[j][0]}:{insts[j][1]}" for j in d)))
                many = len(draws) > 1
                summary.append(dict(pool=pool, method=method, task=task, k=k, draws=len(draws),
                                    auc_mean=aucs.mean(), auc_sd=aucs.std(ddof=1) if many else np.nan,
                                    median_score_sd=np.median(scores.std(axis=0, ddof=1)) if many else np.nan,
                                    disagreement=disagreement(cls) if many else np.nan))
        s = pd.DataFrame(summary)
        s = s[(s.pool == pool) & (s.task == task) & (s.method == "average")]
        print(f"{pool:20s} {task:10s} average: AUC K=1 {s.auc_mean.iloc[0]:.3f} -> "
              f"K={s.k.iloc[-1]} {s.auc_mean.iloc[-1]:.3f}", flush=True)

pd.DataFrame(rows).to_csv(out / "draws.csv", index=False, float_format="%.6g")
summary = pd.DataFrame(summary)
summary.to_csv(out / "summary.csv", index=False, float_format="%.6g")

# AUC against K, one panel per task, one line per pool (averaged predictions).
fig, axes = plt.subplots(1, len(TASKS), figsize=(4.4 * len(TASKS), 4), sharex=True, squeeze=False)
axes = axes[0]
for ax, task in zip(axes, TASKS):
    for pool, g in summary[(summary.task == task) & (summary.method == "average")].groupby("pool"):
        ax.plot(g.k, g.auc_mean, label=pool, lw=2 if pool in PRETRAINED + ["pretrained_mix"] else 1)
    ax.axhline(0.5, color="grey", ls="--", lw=0.8)
    ax.set_title(task)
    ax.set_xlabel("K (instances in the ensemble)")
axes[0].set_ylabel("AUC (mean over draws), averaged predictions")
axes[-1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(out / "curves.png", dpi=150)
print(f"wrote {out}")
