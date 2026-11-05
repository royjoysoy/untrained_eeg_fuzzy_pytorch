"""Rank the extractors of each task, with the uncertainty of the ranking (bootstrap).

Each extractor is scored by its median-instance AUC, its mean AUC over instances and its
ensemble AUC (averaged predictions of all its instances). A resample draws participants with replacement (all samples
of a participant together: blocks, trials or sleep epochs) and, for extractors with several
instances, instances with replacement, then ranks the extractors (1 = best, ties share the
mean rank). For each extractor: the rank on the data, its 95% interval over resamples and the
share of resamples where it ranks first.

Uses the saved out-of-fold predictions (results/phase1/<dataset>/predictions.csv.gz, fixed CV
split), so the probe is not refit in each resample: probe variability is left out and the
intervals are somewhat too narrow. Single-instance extractors (foundation models) have no
instance resampling. Sleep staging uses the macro one-vs-rest AUC.

Writes results/stats/ranking_<dataset>.csv.
    python scripts/ranking.py --config configs/ds004504.yaml    (a few minutes; sleep staging longer)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.data import load_config
from untrained_eeg.foundation import EXCLUDED

STATS = ("median", "mean", "ensemble")


def auc(y, p):
    """Rank-based AUC (equal to sklearn's); macro one-vs-rest when p has one column per class.
    Duplicated samples (bootstrap) share tied ranks."""
    if p.ndim == 2:
        return np.mean([auc((y == k).astype(int), p[:, k]) for k in range(p.shape[1])])
    r = rankdata(p)
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def scores(P, y, idx, inst):
    """Median-instance, mean-instance and ensemble AUC of each extractor (columns in STATS order)
    on samples idx and instances inst."""
    out = np.empty((len(P), len(STATS)))
    for j, (probs, i) in enumerate(zip(P, inst)):
        yb, sub = y[idx], probs[i][:, idx]
        aucs = [auc(yb, p) for p in sub]
        out[j] = np.median(aucs), np.mean(aucs), auc(yb, sub.mean(axis=0))
    return out


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default=None)
parser.add_argument("--resamples", type=int, default=1000)
args = parser.parse_args()
cfg = load_config(args.config) if args.config else load_config()
ds = cfg["openneuro"]["id"]
pred = pd.read_csv(Path(cfg["results_dir"]) / "phase1" / ds / "predictions.csv.gz")
pred = pred[(pred.cv_seed == cfg["cv_seed"]) & ~pred.extractor.isin(EXCLUDED)]  # no BENDR
prob_cols = [c for c in pred.columns if c.startswith("prob")]  # prob, or prob_<k> per class
rng = np.random.default_rng(0)
rows = []
for task, g in pred.groupby("task"):
    samples = g[g.extractor == g.extractor.iloc[0]].drop_duplicates("sample")
    order = {s: i for i, s in enumerate(samples["sample"])}
    y = samples.label.to_numpy()
    extractors = sorted(g.extractor.unique())
    P = []  # per extractor: instances x samples (x classes)
    for e in extractors:
        h = g[g.extractor == e].assign(i=lambda d: d["sample"].map(order)).sort_values(["instance", "i"])
        p = h[prob_cols].to_numpy().reshape(h.instance.nunique(), len(y), len(prob_cols))
        P.append(p[..., 0] if len(prob_cols) == 1 else p)
    clusters = samples["sample"].str.split("/").str[0].to_numpy()  # participant or subject
    members = [np.flatnonzero(clusters == c) for c in np.unique(clusters)]
    all_inst = [np.arange(len(p)) for p in P]
    observed = scores(P, y, np.arange(len(y)), all_inst)
    ranks = np.empty((args.resamples, len(P), len(STATS)))
    for b in range(args.resamples):
        idx = np.concatenate([members[c] for c in rng.integers(0, len(members), len(members))])
        inst = [rng.integers(0, len(p), len(p)) if len(p) > 1 else a for p, a in zip(P, all_inst)]
        ranks[b] = rankdata(-scores(P, y, idx, inst), axis=0)
    point = rankdata(-observed, axis=0)
    for s, stat in enumerate(STATS):
        for j, e in enumerate(extractors):
            r = ranks[:, j, s]
            rows.append(dict(dataset=ds, task=task, statistic=stat, extractor=e, instances=len(P[j]),
                             auc=observed[j, s], rank=point[j, s], rank_lo=np.percentile(r, 2.5),
                             rank_hi=np.percentile(r, 97.5), p_first=(r == 1).mean(), resamples=args.resamples))
    print(f"{task}: {len(extractors)} extractors, {len(members)} participants", flush=True)

out = pd.DataFrame(rows).sort_values(["task", "statistic", "rank"])
dest = Path(cfg["results_dir"]) / "stats" / f"ranking_{ds}.csv"
out.to_csv(dest, index=False, float_format="%.4g")
print(f"wrote {dest}")
print(out[out["rank"] <= 3].to_string(index=False, float_format="%.3f"))
