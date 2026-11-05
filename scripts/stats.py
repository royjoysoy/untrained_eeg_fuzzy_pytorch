"""Significance tests for the model comparisons, from saved phase 1 and phase 2 results.

For every dataset and task:
  above_chance   is the extractor's ensemble (averaged predictions of all its instances)
                 better than chance? Label permutation test of its AUC (10,000 permutations;
                 for eyes closed vs open, labels are permuted within each participant). A quick
                 check only: the probes are not refit on the shuffled labels, so it is not a
                 valid cross-validated permutation test and the paper does not use it
                 (scripts/permutation.py is the primary test)
  vs_matched     pretrained checkpoints vs the untrained network of the same architecture and
                 input: two-sided Mann-Whitney U test between the checkpoints' AUCs and the
                 seeds' AUCs (is the checkpoint distinguishable from a random draw?)
  fm_vs_ensemble each foundation model (one instance) vs the ensemble of untrained
                 ShallowFBCSPNet and of scattering (averaged predictions), on the same samples:
                 paired permutation test of the AUC difference (scores swapped per sample, or
                 per participant where a participant has several samples: all of them together)
Scattering is not tested against the other extractors: its scales J = 3..8 are a design choice,
not a random sample of instances, so it is compared descriptively (its AUC range over J).
Multi-class tasks (sleep staging) use the macro one-vs-rest AUC.

Extractors in foundation.EXCLUDED (BENDR) are left out of every test.
Correction within each family (test x task): max-T (Westfall-Young step-down min-P, p_maxt) for
the permutation tests, whose permutations are shared by every test of the family (the same
shuffled labels for above_chance, the same swaps for fm_vs_ensemble); the Mann-Whitney tests have
no joint permutation null, so they keep Holm (p_holm, also given for every test). Instance AUCs share the same samples, so
these tests ask whether weights (random, pretrained or wavelets) change the result beyond the
weight-to-weight spread; they do not generalize over new participants.

fm_vs_ensemble compares one FM instance with a 30-instance ensemble, which favours the ensemble
by construction, so results/stats/fm_vs_seeds.csv also gives (no test) each FM's AUC against the
single instances of every untrained network and of scattering (not only the two tested
references): their mean, SD and the share it beats (ties count half), next to the ensemble's AUC.

Writes results/stats/stats.csv and fm_vs_seeds.csv, for the datasets of the paper (DATASETS;
--datasets for others found in results/phase1/).

    python scripts/stats.py
"""

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.data import REPO, load_config
from untrained_eeg.foundation import EXCLUDED, FOUNDATION
from untrained_eeg.multitest import holm, maxt
from untrained_eeg.tasks import MULTICLASS, WITHIN_SUBJECT, auc_score

DATASETS = ["ds003478", "ds004504", "ds002778", "lemon", "bnci2014_001", "sleep_edf"]  # the paper's
MATCHED = [("eegnetv4", "eegnet_3ch"), ("shallow_pretrained", "shallow_bcic2a")]
N_PERM = 10_000  # above_chance
N_PAIRED = 10_000  # fm_vs_ensemble (2,000 left about +-0.005 of Monte Carlo error near p = 0.05)


def label_nulls(y, scores, groups, rng):
    """AUC of each score vector and its nulls under the SAME label permutations (within groups if
    given): (observed (m,), null (N_PERM x m)), the joint nulls max-T needs."""
    obs = np.array([auc_score(y, s) for s in scores])
    idx = [np.flatnonzero(groups == g) for g in np.unique(groups)] if groups is not None else None
    null = np.empty((N_PERM, len(scores)))
    for k in range(N_PERM):
        if idx is None:
            yp = rng.permutation(y)
        else:
            yp = y.copy()
            for i in idx:
                yp[i] = rng.permutation(y[i])
        null[k] = [auc_score(yp, s) for s in scores]
    return obs, null


def paired_nulls(y, pairs, groups, rng):
    """|AUC(a) - AUC(b)| for each pair (a, b) of score vectors and its nulls, swapping a and b per
    sample (per group if given: a group's samples are swapped together) with the SAME random swaps
    for every pair: (observed (m,), null (N_PAIRED x m))."""
    obs = np.array([abs(auc_score(y, a) - auc_score(y, b)) for a, b in pairs])
    member = np.unique(groups, return_inverse=True)[1] if groups is not None else np.arange(len(y))
    null = np.empty((N_PAIRED, len(pairs)))
    for k in range(N_PAIRED):
        swap = (rng.random(member.max() + 1) < 0.5)[member]
        for j, (a, b) in enumerate(pairs):
            sw = swap[:, None] if a.ndim == 2 else swap
            null[k, j] = abs(auc_score(y, np.where(sw, b, a)) - auc_score(y, np.where(sw, a, b)))
    return obs, null


def raw_p(obs, null):
    """Permutation p-value of each column: (1 + #null >= observed) / (1 + n)."""
    return (1 + (null >= obs).sum(axis=0)) / (1 + len(null))


def ensemble_scores(g):
    """Averaged predictions of an extractor's instances: (sample ids, labels, scores)."""
    cols = [c for c in g.columns if c.startswith("prob_") and g[c].notna().any()] or ["prob"]
    w = [g.pivot_table(index="sample", columns="instance", values=c) for c in cols]
    s = np.stack([x.mean(axis=1).to_numpy() for x in w], axis=1)
    y = g.drop_duplicates("sample").set_index("sample").label.loc[w[0].index].to_numpy()
    return w[0].index, y, (s if s.shape[1] > 1 else s[:, 0]), w[0].shape[1]


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--datasets", nargs="+", default=DATASETS, help="folders of results/phase1/ to test")
args = parser.parse_args()

CONFIGS = [load_config(p) for p in sorted((REPO / "configs").glob("*.yaml"))]
CV_SEED = {c["openneuro"]["id"]: c["cv_seed"] for c in CONFIGS}  # each dataset's fixed split
rows, seed_rows = [], []
for d in [REPO / "results" / "phase1" / ds for ds in args.datasets]:
    if not (d / "metrics.csv").exists():
        continue
    ds = d.name
    m = pd.read_csv(d / "metrics.csv")
    m = m[(m.cv_seed == CV_SEED[ds]) & ~m.extractor.isin(EXCLUDED)]  # the config's fixed split; no BENDR
    preds = pd.read_csv(d / "predictions.csv.gz")
    preds = preds[(preds.cv_seed == CV_SEED[ds]) & ~preds.extractor.isin(EXCLUDED)]
    for task, mt in m.groupby("task"):
        # one random stream per task: its permutations do not depend on the other datasets run
        rng = np.random.default_rng(zlib.crc32(f"{ds}|{task}".encode()))
        auc = {e: g.auc.to_numpy() for e, g in mt.groupby("extractor")}
        pt = preds[preds.task == task]
        ens = {e: ensemble_scores(g) for e, g in pt.groupby("extractor")}  # all instances averaged
        ids, y = next(iter(ens.values()))[:2]
        assert all(list(v[0]) == list(ids) for v in ens.values()), "extractors must share the samples"
        # ensemble of all instances vs chance, every extractor on the same label permutations
        clustered = task == "ec_eo" or task in WITHIN_SUBJECT or task in MULTICLASS  # several samples per person
        groups = np.array([i.split("/")[0] for i in ids]) if clustered else None
        obs, null = label_nulls(y, [v[2] for v in ens.values()], groups, rng)
        for (e, (_, _, _, n)), o, p, pm in zip(ens.items(), obs, raw_p(obs, null), maxt(null, obs)):
            rows.append(dict(test="above_chance", dataset=ds, task=task, a=e, b="chance",
                             auc_a=o, auc_b=0.5, n_a=n, n_b=0, p=p, p_maxt=pm))
        pairs = [(fm, ref) for fm in FOUNDATION if fm in ens for ref in ("shallow", "scattering") if ref in ens]
        if pairs:  # each FM vs the ensembles, every pair on the same random swaps
            obs, null = paired_nulls(y, [(ens[a][2], ens[b][2]) for a, b in pairs], groups, rng)
            for (a, b), p, pm in zip(pairs, raw_p(obs, null), maxt(null, obs)):
                rows.append(dict(test="fm_vs_ensemble", dataset=ds, task=task, a=a, b=b,
                                 auc_a=auc_score(y, ens[a][2]), auc_b=auc_score(y, ens[b][2]),
                                 n_a=1, n_b=ens[b][3], p=p, p_maxt=pm))
        refs = sorted(mt[mt.group.isin(["untrained", "wavelet"])].extractor.unique())
        for fm in [e for e in FOUNDATION if e in ens]:
            for ref in refs:
                if ref in ens:  # descriptive: the FM against single instances, then the ensemble
                    v = auc[ref]
                    beaten = ((auc[fm][0] > v).sum() + 0.5 * (auc[fm][0] == v).sum()) / len(v)
                    seed_rows.append(dict(dataset=ds, task=task, fm=fm, ref=ref, fm_auc=auc[fm][0],
                                          ref_mean=v.mean(), ref_sd=v.std(ddof=1), ref_instances=len(v),
                                          share_beaten=beaten, ref_ensemble_auc=auc_score(*ens[ref][1:3])))
        for pre, unt in MATCHED:
            if pre in auc and unt in auc:
                p = mannwhitneyu(auc[pre], auc[unt], alternative="two-sided").pvalue
                rows.append(dict(test="vs_matched", dataset=ds, task=task, a=pre, b=unt, auc_a=auc[pre].mean(),
                                 auc_b=auc[unt].mean(), n_a=len(auc[pre]), n_b=len(auc[unt]), p=p))
    print(f"{ds}: {sum(r['dataset'] == ds for r in rows)} tests", flush=True)

s = pd.DataFrame(rows)
s["p_holm"] = s.groupby(["test", "dataset", "task"]).p.transform(holm)  # per task
# max-T where the test has joint permutation nulls; Holm for the Mann-Whitney tests
s["significant"] = s.p_maxt.fillna(s.p_holm) < 0.05
s["key"] = s.dataset + "|" + s.test + "|" + s.task + "|" + s.a + "|" + s.b
out = REPO / "results" / "stats"
out.mkdir(parents=True, exist_ok=True)
s.to_csv(out / "stats.csv", index=False, float_format="%.6g")
pd.DataFrame(seed_rows).to_csv(out / "fm_vs_seeds.csv", index=False, float_format="%.6g")
print(s.groupby(["test", "dataset"]).significant.agg(["sum", "count"]).to_string())
print(f"wrote {out / 'stats.csv'}")
