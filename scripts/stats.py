"""Significance tests for the model comparisons, from saved phase 1 and phase 2 results.

For every dataset and task:
  above_chance   is the extractor's ensemble (averaged predictions of all its instances)
                 better than chance? Label permutation test of its AUC (10,000 permutations;
                 for eyes closed vs open, labels are permuted within each participant)
  vs_matched     pretrained checkpoints vs the untrained network of the same architecture and
                 input: two-sided Mann-Whitney U test between the checkpoints' AUCs and the
                 seeds' AUCs (is the checkpoint distinguishable from a random draw?)
  vs_scattering  wavelet scattering (one AUC per scale J) vs each other extractor (one AUC per
                 seed or checkpoint), Mann-Whitney U (only extractors with 3+ instances)
  fm_vs_ensemble each foundation model (one instance) vs the ensemble of untrained
                 ShallowFBCSPNet and of scattering (averaged predictions), on the same samples:
                 paired permutation test of the AUC difference (scores swapped per sample)
Multi-class tasks (sleep staging) use the macro one-vs-rest AUC.

Holm correction within each family (test x dataset). Instance AUCs share the same samples, so
these tests ask whether weights (random, pretrained or wavelets) change the result beyond the
weight-to-weight spread; they do not generalize over new participants.

Writes results/stats/stats.csv.

    python scripts/stats.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.data import REPO
from untrained_eeg.foundation import FOUNDATION
from untrained_eeg.tasks import MULTICLASS, auc_score

MATCHED = [("eegnetv4", "eegnet_3ch"), ("shallow_pretrained", "shallow_bcic2a")]
N_PERM = 10_000


def holm(p):
    """Holm-adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def permutation_p(y, score, groups, rng):
    """One-sided p-value of AUC(y, score) under label permutations (within groups if given)."""
    obs = auc_score(y, score)
    count = 0
    idx = [np.flatnonzero(groups == g) for g in np.unique(groups)] if groups is not None else None
    for _ in range(N_PERM):
        if idx is None:
            yp = rng.permutation(y)
        else:
            yp = y.copy()
            for i in idx:
                yp[i] = rng.permutation(y[i])
        count += auc_score(yp, score) >= obs
    return obs, (count + 1) / (N_PERM + 1)


def paired_p(y, a, b, rng, n_perm=2000):
    """Two-sided p-value of AUC(a) - AUC(b), swapping a and b per sample at random."""
    obs = auc_score(y, a) - auc_score(y, b)
    count = 0
    for _ in range(n_perm):
        swap = rng.random(len(y)) < 0.5
        sw = swap[:, None] if a.ndim == 2 else swap
        count += abs(auc_score(y, np.where(sw, b, a)) - auc_score(y, np.where(sw, a, b))) >= abs(obs)
    return obs, (count + 1) / (n_perm + 1)


def ensemble_scores(g):
    """Averaged predictions of an extractor's instances: (sample ids, labels, scores)."""
    cols = [c for c in g.columns if c.startswith("prob_") and g[c].notna().any()] or ["prob"]
    w = [g.pivot_table(index="sample", columns="instance", values=c) for c in cols]
    s = np.stack([x.mean(axis=1).to_numpy() for x in w], axis=1)
    y = g.drop_duplicates("sample").set_index("sample").label.loc[w[0].index].to_numpy()
    return w[0].index, y, (s if s.shape[1] > 1 else s[:, 0]), w[0].shape[1]


rng = np.random.default_rng(0)
rows = []
for d in sorted((REPO / "results" / "phase1").iterdir()):
    if not (d / "metrics.csv").exists():
        continue
    ds = d.name
    m = pd.read_csv(d / "metrics.csv")
    m = m[m.cv_seed == m.cv_seed.min()]  # the fixed split of the config
    preds = pd.read_csv(d / "predictions.csv.gz")
    preds = preds[preds.cv_seed == preds.cv_seed.min()]
    for task, mt in m.groupby("task"):
        auc = {e: g.auc.to_numpy() for e, g in mt.groupby("extractor")}
        pt = preds[preds.task == task]
        ens = {}
        for e, g in pt.groupby("extractor"):  # ensemble of all instances vs chance
            ids, y, s, n = ens[e] = ensemble_scores(g)
            clustered = task in ("ec_eo", "mi_left_right") or task in MULTICLASS  # several samples per person
            groups = np.array([i.split("/")[0] for i in ids]) if clustered else None
            obs, p = permutation_p(y, s, groups, rng)
            rows.append(dict(test="above_chance", dataset=ds, task=task, a=e, b="chance",
                             auc_a=obs, auc_b=0.5, n_a=n, n_b=0, p=p))
        for fm in [e for e in FOUNDATION if e in ens]:
            for ref in ("shallow", "scattering"):
                if ref in ens:
                    ids, y, s_fm, _ = ens[fm]
                    s_ref = ens[ref][2][pd.Index(ens[ref][0]).get_indexer(ids)]
                    diff, p = paired_p(y, s_fm, s_ref, rng)
                    rows.append(dict(test="fm_vs_ensemble", dataset=ds, task=task, a=fm, b=ref,
                                     auc_a=auc_score(y, s_fm), auc_b=auc_score(y, s_ref), n_a=1, n_b=ens[ref][3], p=p))
        for pre, unt in MATCHED:
            if pre in auc and unt in auc:
                p = mannwhitneyu(auc[pre], auc[unt], alternative="two-sided").pvalue
                rows.append(dict(test="vs_matched", dataset=ds, task=task, a=pre, b=unt, auc_a=auc[pre].mean(),
                                 auc_b=auc[unt].mean(), n_a=len(auc[pre]), n_b=len(auc[unt]), p=p))
        if "scattering" in auc:
            for e, v in auc.items():
                if e != "scattering" and len(v) >= 3:
                    p = mannwhitneyu(auc["scattering"], v, alternative="two-sided").pvalue
                    rows.append(dict(test="vs_scattering", dataset=ds, task=task, a="scattering", b=e,
                                     auc_a=auc["scattering"].mean(), auc_b=v.mean(),
                                     n_a=len(auc["scattering"]), n_b=len(v), p=p))
    print(f"{ds}: {sum(r['dataset'] == ds for r in rows)} tests", flush=True)

s = pd.DataFrame(rows)
s["p_holm"] = s.groupby(["test", "dataset"]).p.transform(holm)
s["significant"] = s.p_holm < 0.05
s["key"] = s.dataset + "|" + s.test + "|" + s.task + "|" + s.a + "|" + s.b
out = REPO / "results" / "stats"
out.mkdir(parents=True, exist_ok=True)
s.to_csv(out / "stats.csv", index=False, float_format="%.6g")
print(s.groupby(["test", "dataset"]).significant.agg(["sum", "count"]).to_string())
print(f"wrote {out / 'stats.csv'}")
