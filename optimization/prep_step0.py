#!/usr/bin/env python
"""Step (b), step 0: the reference level must reproduce step (a) exactly.

    python optimization/prep_step0.py --dataset ds003478 --data-root $S/data --out $OUT \
        --step-a $A/11-5-26 $A/11-5-26b --seeds 0-2 [--check-levels]

1. For every architecture extracted at <out>/prep/ref/<dataset>/features/cnn/<arch>/ and each
   seed: features, block rows and dtype equal the step-(a) file (first --step-a folder that
   has it), bit for bit; c1 AUC and balanced accuracy (float32 features as stored) equal the
   step-(a) per_seed.csv values. Handcrafted families too, if extracted at the reference.
2. --check-levels: every level that applies to the dataset gives the same blocks (participants
   and conditions, in order) as the reference, so all levels share samples and folds. Builds
   nothing: raw-based caches not built yet are reported as missing.
Exits 1 on any mismatch.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimization import prep  # noqa: E402
from optimization.common import block_features, dataset_config  # noqa: E402
from optimization.evaluate import c1  # noqa: E402
from untrained_eeg.tasks import task_data, task_labels, task_names  # noqa: E402


def seed_range(text):
    a, _, b = text.partition("-")
    return list(range(int(a), int(b or a) + 1))


def same_file(ours, theirs):
    A, B = np.load(ours), np.load(theirs)
    # equal_nan: a failed fit (e.g. FOOOF's aperiodic) stores NaN, and NaN != NaN otherwise
    return (A["feats"].dtype == B["feats"].dtype and np.array_equal(A["feats"], B["feats"], equal_nan=True)
            and np.array_equal(A["participant_id"], B["participant_id"]) and np.array_equal(A["cond"], B["cond"]))


def block_rows(windows, per_block, keep):
    """(participant, condition) of every block, in order."""
    rows = []
    for p, (X, cond) in windows.items():
        _, c = block_features(np.zeros((len(X), 1), np.float32), cond, per_block,
                              None if keep is None else keep[p])
        rows += [(p, x) for x in c]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--step-a", type=Path, nargs="+", required=True)
    ap.add_argument("--seeds", type=seed_range, default=[0, 1, 2])
    ap.add_argument("--cache-root", type=Path, default=None)
    ap.add_argument("--check-levels", action="store_true")
    a = ap.parse_args()
    cfg = dataset_config(a.dataset, a.data_root)
    labels, tasks = task_labels(cfg), task_names(cfg)
    ours_root = a.out / "prep" / "ref" / a.dataset / "features"
    ok, rows = True, []

    def find(rel):
        return next((d / a.dataset / "features" / rel for d in a.step_a if (d / a.dataset / "features" / rel).exists()), None)

    step_a = pd.concat([pd.read_csv(f) for d in a.step_a for f in [d / a.dataset / "per_seed.csv",
                        d / a.dataset / "per_seed_search.csv"] if f.exists()], ignore_index=True)
    for arch_dir in sorted((ours_root / "cnn").glob("*")):
        for s in a.seeds:
            ours = arch_dir / f"seed{s:03d}.npz"
            theirs = find(Path("cnn") / arch_dir.name / ours.name)
            if not ours.exists() or theirs is None:
                print(f"missing: {ours if not ours.exists() else 'step (a) ' + arch_dir.name + ' ' + ours.name}")
                ok = False
                continue
            bits = same_file(ours, theirs)
            f = np.load(ours)
            for t in tasks:
                X, y, g, _ = task_data(f["feats"], f["cond"], f["participant_id"], t, labels)
                _, auc, bacc = c1(X, y, g, t, cfg)
                r = step_a[(step_a.config == arch_dir.name) & (step_a.seed == s) & (step_a.task == t)]
                match = len(r) > 0 and abs(r.auc.iloc[0] - auc) < 5e-6 and abs(r.balanced_accuracy.iloc[0] - bacc) < 5e-6
                rows.append(dict(arch=arch_dir.name, seed=s, task=t, bit_identical=bits, auc=auc,
                                 step_a_auc=r.auc.iloc[0] if len(r) else np.nan, metrics_match=match))
                ok &= bits and match
    for f in sorted((ours_root / "handcrafted").glob("*.npz")):
        theirs = find(Path("handcrafted") / f.name)
        bits = theirs is not None and same_file(f, theirs)
        rows.append(dict(arch=f"handcrafted:{f.stem}", bit_identical=bits))
        ok &= bits
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(6).to_string(index=False))

    if a.check_levels:
        _, _, W, b, K = prep.load(a.dataset, a.data_root, "ref")
        ref = block_rows(W, b, K)
        del W
        for lv in prep.LEVELS:
            if lv == "ref" or prep.LEVELS[lv].kernel_x2 or not prep.applies(lv, cfg):
                continue
            try:
                _, _, W, b, K = prep.load(a.dataset, a.data_root, lv, a.cache_root)
            except SystemExit as e:  # e.g. a raw-based cache not built yet
                print(f"{lv:14s} not checked: {e}")
                continue
            same = block_rows(W, b, K) == ref
            extra = prep.rejection_stats(K)
            print(f"{lv:14s} same blocks as ref: {same}  {extra}")
            ok &= same
            del W
    print("\nSTEP 0 (b):", "MATCH" if ok else "MISMATCH - stop and investigate")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    # one BLAS thread, as in step (a)'s probes (joblib workers get one thread each): the thread
    # count changes summation order, which moves e.g. the 5-class AUC in the 5th decimal
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        main()
