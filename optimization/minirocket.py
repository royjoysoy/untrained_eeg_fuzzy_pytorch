#!/usr/bin/env python
"""MiniRocket baseline (Dempster et al., 2021) with the same blocks, splits and probe.

    python optimization/minirocket.py --dataset ds003478 --data-root $S/data \
        --out results/optimization/11-5-26 --seeds 0-9 [--jobs 8]

MiniRocket's kernels are fixed, but its biases are quantiles of convolution outputs
on example series, i.e. learned from data. So for every outer split of every task:
  1. fit MiniRocket (biases) on windows of the TRAINING samples only,
  2. transform every window of the samples in this split, average windows per block
     (as phase 1), then untrained_eeg.tasks.task_data builds the task's samples,
  3. the c1 probe (standardize + logistic regression, C = 1) is fitted on the training
     samples and scores the test samples.
Outer splits are untrained_eeg.tasks._outer_splits with the config's cv_seed (the same
as every other extractor). random_state (channel combinations, example choice) = seed.
Writes <out>/<dataset>/minirocket.csv with per_seed.csv's columns (kind = minirocket).
Step (b): --prep <level> reads that preprocessing level (optimization/prep.py) and writes
under <out>/prep/<level>/ (artifact windows left out of the block means, median / trimmed
aggregation applied to the window features, as for the CNN).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimization import prep  # noqa: E402
from optimization.common import aggregate, dataset_config, load_windows  # noqa: E402
from optimization.evaluate import _finish, estimator  # noqa: E402
from optimization.runinfo import run_info  # noqa: E402
from untrained_eeg.tasks import MULTICLASS, WITHIN_SUBJECT, _outer_splits, task_data, task_labels, task_names  # noqa: E402

BLOCK_LEVEL = set(MULTICLASS) | WITHIN_SUBJECT | {"ec_eo"}  # one sample per block, not per participant


def seed_range(text):
    a, _, b = text.partition("-")
    return list(range(int(a), int(b or a) + 1))


class Blocks:
    """Index from blocks to their windows, for all participants."""

    def __init__(self, participants, windows, per_block, keep=None, agg="mean"):
        self.W, self.pid, self.cond, self.start = [], [], [], []
        self.keep = None if keep is None else np.concatenate(
            [keep[p][: len(windows[p][0]) // per_block * per_block] for p in participants["participant_id"]])
        self.agg = agg
        off = 0
        for p in participants["participant_id"]:
            X, c = windows[p]
            n = len(X) // per_block
            self.W.append(X[: n * per_block])
            self.pid += [p] * n
            self.cond += list(c[::per_block][:n])
            self.start += [off + i * per_block for i in range(n)]
            off += n * per_block
        self.W = np.concatenate(self.W).astype(np.float32)
        self.pid, self.cond, self.start = map(np.asarray, (self.pid, self.cond, self.start))
        self.per_block = per_block

    def windows_of(self, blocks):
        return np.concatenate([np.arange(s, s + self.per_block) for s in self.start[blocks]])


def block_means(model, B, blocks, chunk=2000):
    """MiniRocket features of the given blocks: mean over each block's windows."""
    step = max(1, chunk // B.per_block)
    rows = []
    for i in range(0, len(blocks), step):
        b = blocks[i: i + step]
        w = B.windows_of(b)
        F = np.asarray(model.transform(B.W[w]), np.float32).reshape(len(b), B.per_block, -1)
        if B.keep is None and B.agg == "mean":
            rows.append(F.mean(axis=1))  # the step-(a) code path
        else:
            rows.append(aggregate(F, None if B.keep is None else B.keep[w], B.agg))
    return np.concatenate(rows)


def run_task(B, task, labels, cfg, seed, n_jobs, max_fit_windows):
    from aeon.transformations.collection.convolution_based import MiniRocket
    n_blocks = len(B.pid)
    X0, y, groups, _ = task_data(np.arange(n_blocks, dtype=float)[:, None], B.cond, B.pid, task, labels)
    rng = np.random.default_rng(seed)
    n_cls = len(np.unique(y))
    prob = np.zeros((len(y), n_cls))

    def blocks_of(samples):
        if task in BLOCK_LEVEL:
            return np.unique(X0[samples, 0].astype(int))
        return np.flatnonzero(np.isin(B.pid, groups[samples]))

    for tr, te in _outer_splits(X0, y, groups, task, cfg["cv_seed"], cfg["cv_folds"]):
        fit_w = B.windows_of(blocks_of(tr))
        if len(fit_w) > max_fit_windows:
            fit_w = np.sort(rng.choice(fit_w, max_fit_windows, replace=False))
        model = MiniRocket(n_kernels=10000, random_state=seed, n_jobs=n_jobs).fit(B.W[fit_w])
        used = blocks_of(np.concatenate([tr, te]))
        F = block_means(model, B, used)
        feats = np.zeros((n_blocks, F.shape[1]), np.float32)
        feats[used] = F
        X, y2, _, _ = task_data(feats, B.cond, B.pid, task, labels)
        assert np.array_equal(y2, y)
        prob[te] = estimator().fit(X[tr], y[tr]).predict_proba(X[te])
    return _finish(y, prob, task), F.shape[1], len(y)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=seed_range, default=list(range(10)))
    p.add_argument("--max-fit-windows", type=int, default=5000,
                   help="training windows sampled to fit the biases (MiniRocket only needs examples)")
    p.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    p.add_argument("--prep", choices=list(prep.LEVELS), default=None, help="step (b) preprocessing level")
    p.add_argument("--cache-root", type=Path, default=None, help="where raw-based level caches live")
    a = p.parse_args()
    if a.prep and not prep.for_minirocket(a.prep):
        print(f"{a.prep}: MiniRocket input is the same as another level's; nothing to do")
        return
    run_out = a.out
    if a.prep:
        a.out = a.out / "prep" / a.prep
    dest = a.out / a.dataset / "minirocket.csv"
    if dest.exists():
        raise FileExistsError(dest)
    with run_info(run_out, "minirocket", a.dataset, extra=dict(seeds=a.seeds, prep=a.prep)):
        _main(a, dest)


def _main(a, dest):
    if a.prep:
        cfg, participants, windows, per_block, keep = prep.load(a.dataset, a.data_root, a.prep, a.cache_root)
        B = Blocks(participants, windows, per_block, keep, prep.LEVELS[a.prep].agg)
    else:
        cfg = dataset_config(a.dataset, a.data_root)
        participants, windows, per_block = load_windows(cfg)
        B = Blocks(participants, windows, per_block)
    fac, val = (prep.LEVELS[a.prep].factor, a.prep) if a.prep else ("baseline", "minirocket")
    labels, tasks = task_labels(cfg), task_names(cfg)
    rows = []
    for seed in a.seeds:
        for task in tasks:
            t0 = time.time()
            (_, auc, bacc), dim, n = run_task(B, task, labels, cfg, seed, a.jobs, a.max_fit_windows)
            rows.append(dict(dataset=a.dataset, task=task, kind="minirocket", factor=fac,
                             value=val, config="minirocket", seed=seed, n=n, auc=auc,
                             balanced_accuracy=bacc, feature_dim=dim, receptive_field_s=np.nan,
                             extract_seconds=time.time() - t0, n_windows=len(B.W), n_jobs=a.jobs))
            print(f"{a.dataset} {task:14s} seed{seed} AUC {auc:.3f} bacc {bacc:.3f} "
                  f"({rows[-1]['extract_seconds']:.0f}s)", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest, index=False, float_format="%.6g")
    print("wrote", dest)


if __name__ == "__main__":
    main()
