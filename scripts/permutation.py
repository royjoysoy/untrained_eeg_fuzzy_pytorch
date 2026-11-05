"""Permutation test of every model on every task of one dataset (refitting the probe).

For each extractor and task: the observed AUC against the AUCs obtained after shuffling the
labels and refitting the same probe with the same CV split (Ojala and Garriga 2010;
scikit-learn's permutation_test_score). Labels are shuffled within each participant where a
participant has several samples (eyes closed vs open, motor imagery, sleep staging), freely
otherwise. Multiple-testing correction is per task, over its extractors (and per statistic).

Two modes:
  default     the first instance only (seed000, the first checkpoint, scattering's J3 or the
              single foundation model, fixed in advance); p = (1 + #null >= observed) / (1 + n)
  --ensemble  ALL instances of the extractor. Every instance's probe is refit on the same
              shuffled labels, and three statistics are taken from these probes, each compared
              with the same statistic under shuffled labels:
                ensemble  AUC of the averaged out-of-fold predictions (as in phase 2)
                median    median AUC over instances: does a typical instance beat chance?
                best      best AUC over instances: does any instance, picked after the fact?
                          (a max-statistic test, so picking the best is accounted for)
              Single-instance extractors (foundation models) get the same value three times.
              Each statistic stops counting once 20 null values reach the observed one
              (Besag and Clifford 1991: p = 20 / n); permutations stop when all three have,
              so only tests that may be significant run all --permutations.
  --maxt      no early stop, every extractor runs all --permutations, and the null values
              are kept (results/stats/nulls_<dataset>/, nulls_first_<dataset>/ for the first
              instance; output permutation_first_maxt_<dataset>.csv). Every extractor sees
              the same shuffled labels at permutation i, so the nulls are joint and the
              Westfall-Young step-down min-P correction (column p_maxt) can use the correlation
              between extractors. Like Holm it controls the familywise error over a task's
              extractors, with more power when their results are correlated.

Correction: p_holm (Holm) and, with --maxt, p_maxt, each per task and statistic;
significant = p_maxt < 0.05 with --maxt, else p_holm < 0.05. Extractors in foundation.EXCLUDED
(BENDR) are not tested. --correct-only recomputes the correction of an existing output file
without them (with --maxt, from the saved nulls), without running any permutation.

Writes results/stats/permutation_<dataset>.csv (default), permutation_ensemble_<dataset>.csv or
permutation_maxt_<dataset>.csv (one row per extractor, task and statistic); the ensemble modes
save each finished test, so a job that times out can be resubmitted.
Run it as a job (one probe per instance and permutation):
    CONFIG=configs/ds004504.yaml sbatch slurm/permutation.sh
    CONFIG=configs/ds004504.yaml ARGS="--ensemble --permutations 10000" sbatch slurm/permutation.sh
    CONFIG=configs/ds004504.yaml ARGS="--ensemble --maxt --permutations 10000" sbatch slurm/permutation.sh
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.data import load_config
from untrained_eeg.foundation import EXCLUDED
from untrained_eeg.multitest import holm, maxt
from untrained_eeg.tasks import MULTICLASS, WITHIN_SUBJECT, auc_score, probe, task_data, task_labels, task_names

STOP_AFTER = 20  # ensemble mode: null values >= observed before a statistic stops counting
STATS = ("ensemble", "median", "best")  # ensemble mode: statistics of the instances' probes


def shuffled(y, groups, rng):
    """y with its labels permuted, within each group if groups is given."""
    if groups is None:
        return rng.permutation(y)
    out = y.copy()
    for g in np.unique(groups):
        i = np.flatnonzero(groups == g)
        out[i] = rng.permutation(y[i])
    return out


def instance_stats(Xs, y, groups, task, cfg):
    """Ensemble AUC (averaged out-of-fold predictions), median and best AUC of one probe per instance."""
    probs = [probe(X, y, groups, task, cfg["cv_seed"], cfg["cv_folds"])[0] for X in Xs]
    aucs = [auc_score(y, p) for p in probs]
    return auc_score(y, np.mean(probs, axis=0)), np.median(aucs), np.max(aucs)


def null_stats(Xs, y, groups, task, cfg, seeds):
    """instance_stats after shuffling the labels with each seed (the same labels for every instance)."""
    clustered = task == "ec_eo" or task in WITHIN_SUBJECT or task in MULTICLASS
    return [instance_stats(Xs, shuffled(y, groups if clustered else None, np.random.default_rng(s)), groups, task, cfg)
            for s in seeds]


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default=None)
parser.add_argument("--permutations", type=int, default=None, help="default: config n_permutations, else 1000")
parser.add_argument("--ensemble", action="store_true", help="test all instances: ensemble, median, best (see above)")
parser.add_argument("--maxt", action="store_true", help="no early stop, max-T correction (see above)")
parser.add_argument("--correct-only", action="store_true", help="recompute the correction of the existing output")
parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
args = parser.parse_args()

cfg = load_config(args.config) if args.config else load_config()
n_perm = args.permutations or cfg.get("n_permutations", 1000)
ds = cfg["openneuro"]["id"]
src = Path(cfg["results_dir"]) / "phase1" / ds / "features"
mode = {(True, True): "maxt_", (True, False): "ensemble_", (False, True): "first_maxt_", (False, False): ""}
dest = Path(cfg["results_dir"]) / "stats" / f"permutation_{mode[args.ensemble, args.maxt]}{ds}.csv"
dest.parent.mkdir(parents=True, exist_ok=True)
# --maxt: one (permutations x statistics) array per test
nulls = dest.parent / (f"nulls_{ds}" if args.ensemble else f"nulls_first_{ds}")
if args.maxt:
    nulls.mkdir(exist_ok=True)
labels = task_labels(cfg)
rows = []
partial = dest.with_suffix(".partial.csv")  # tests finished so far
if args.correct_only:  # the finished tests, corrected again below
    rows = pd.read_csv(dest).drop(columns=["p_holm", "p_maxt", "significant"], errors="ignore").to_dict("records")
elif partial.exists():
    rows = pd.read_csv(partial).to_dict("records")
rows = [r for r in rows if r["extractor"] not in EXCLUDED]
done = {(r["extractor"], r["task"]) for r in rows}
ext_dirs = [] if args.correct_only else sorted(p for p in src.iterdir() if p.is_dir() and p.name not in EXCLUDED)

for ext_dir in ext_dirs:
    paths = sorted(ext_dir.glob("*.npz"))
    if not args.ensemble:
        paths = paths[:1]  # the first instance, as in phase1_probe's CV-split runs
    files = [np.load(p) for p in paths]
    for task in task_names(cfg):
        if (ext_dir.name, task) in done:
            continue
        data = [task_data(f["feats"], f["cond"], f["participant_id"], task, labels) for f in files]
        Xs, (_, y, groups, ids) = [d[0] for d in data], data[0]
        assert all(d[3] == ids for d in data), "instances must share the same samples"
        observed = np.array(instance_stats(Xs, y, groups, task, cfg))
        if not args.ensemble:  # one instance: the three statistics are its AUC
            null = np.array(Parallel(n_jobs=args.workers)(
                delayed(null_stats)(Xs, y, groups, task, cfg, [seed]) for seed in range(n_perm)))[:, 0, 0]
            p = (1 + (null >= observed[0]).sum()) / (1 + n_perm)
            if args.maxt:
                np.save(nulls / f"{ext_dir.name}__{task}.npy", null[:, None])
            rows.append(dict(dataset=ds, task=task, extractor=ext_dir.name, statistic="first",
                             instance=paths[0].stem, auc=observed[0], null_mean=null.mean(),
                             null_p95=np.quantile(null, 0.95), n_permutations=n_perm, p=p))
            print(f"{ext_dir.name:18s} {task:15s} AUC {observed[0]:.3f}  null {null.mean():.3f} "
                  f"(95% {np.quantile(null, 0.95):.3f})  p {p:.4f}", flush=True)
            pd.DataFrame(rows).to_csv(partial, index=False, float_format="%.6g")
            continue
        # Ensemble mode: rounds of one chunk of permutations per worker, until done or (without
        # --maxt) every statistic has STOP_AFTER null values >= its observed value.
        null, chunk = np.empty((0, len(STATS))), max(1, min(50, n_perm // (10 * args.workers)))
        stop = not args.maxt
        while len(null) < n_perm and not (stop and ((null >= observed).sum(axis=0) >= STOP_AFTER).all()):
            start = len(null)
            seeds = list(range(start, min(n_perm, start + chunk * args.workers)))
            parts = Parallel(n_jobs=args.workers)(
                delayed(null_stats)(Xs, y, groups, task, cfg, seeds[i : i + chunk])
                for i in range(0, len(seeds), chunk))
            null = np.concatenate([null, *map(np.array, parts)])
        if args.maxt:
            np.save(nulls / f"{ext_dir.name}__{task}.npy", null)
        for i, stat in enumerate(STATS):
            hits = np.cumsum(null[:, i] >= observed[i])
            stopped = stop and hits[-1] >= STOP_AFTER
            n = int(np.searchsorted(hits, STOP_AFTER)) + 1 if stopped else len(null)
            p = STOP_AFTER / n if stopped else (1 + hits[n - 1]) / (1 + n)
            rows.append(dict(dataset=ds, task=task, extractor=ext_dir.name, statistic=stat,
                             instance=f"{len(paths)} instances", auc=observed[i], null_mean=null[:n, i].mean(),
                             null_p95=np.quantile(null[:n, i], 0.95), n_permutations=n, p=p))
            print(f"{ext_dir.name:18s} {task:15s} {stat:8s} AUC {observed[i]:.3f}  null {null[:n, i].mean():.3f} "
                  f"(95% {np.quantile(null[:n, i], 0.95):.3f}, n {n})  p {p:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(partial, index=False, float_format="%.6g")

out = pd.DataFrame(rows)
if "statistic" not in out:  # first-instance outputs written before the column existed
    out["statistic"] = "first"
families = ["statistic", "task"]  # corrected per task and statistic
out["p_holm"] = out.groupby(families).p.transform(holm)
if args.maxt:  # joint nulls of the task's extractors: the same shuffled labels at each permutation
    for (stat, task), g in out.groupby(families):
        col = 0 if stat == "first" else STATS.index(stat)
        null = np.column_stack([np.load(nulls / f"{e}__{task}.npy")[:, col] for e in g.extractor])
        out.loc[g.index, "p_maxt"] = maxt(null, g.auc.to_numpy())
out["significant"] = (out.p_maxt if args.maxt else out.p_holm) < 0.05
out.to_csv(dest, index=False, float_format="%.6g")
if not args.correct_only:
    partial.unlink(missing_ok=True)
method = "max-T" if args.maxt else "Holm"
for key, g in out.groupby("statistic"):
    print(f"wrote {dest}: {key}: {g.significant.sum()} of {len(g)} significant after {method} (per task)")
