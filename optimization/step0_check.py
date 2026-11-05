#!/usr/bin/env python
"""Step 0: reproduce the phase-1 cnn1d baseline before optimizing anything.

    python optimization/step0_check.py --dataset ds003478 --data-root $S/data \
        --phase1 $S/results/phase1 --out results/optimization/11-5-26 [--seeds 0-29]

Checks, and exits 1 on any mismatch:
 1. Our probe on phase 1's OWN cnn1d feature files gives exactly the AUC and balanced
    accuracy in phase1/<dataset>/metrics.csv (cv_seed 0), for every seed and task.
 2. Our default config re-extracted from the cache (<out>/<dataset>/features/cnn/default)
    has the same block rows as phase 1 and matches its features (max relative difference
    reported; bit-identical only with the same PyTorch build), and gives the same metrics.
 3. Multi-seed mean +- SD of AUC per task, ours vs phase 1.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimization.common import dataset_config  # noqa: E402
from optimization.evaluate import c1  # noqa: E402
from untrained_eeg.tasks import task_data, task_labels, task_names  # noqa: E402


def seed_range(text):
    a, _, b = text.partition("-")
    return list(range(int(a), int(b or a) + 1))


def metrics(path, cfg, labels, tasks):
    f = np.load(path)
    out = {}
    for t in tasks:
        X, y, g, _ = task_data(f["feats"].astype(np.float64), f["cond"], f["participant_id"], t, labels)
        _, auc, bacc = c1(X, y, g, t, cfg)
        out[t] = (auc, bacc)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="ds003478")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--phase1", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=seed_range, default=list(range(30)))
    a = p.parse_args()
    cfg = dataset_config(a.dataset, a.data_root)
    labels, tasks = task_labels(cfg), task_names(cfg)
    ref = pd.read_csv(a.phase1 / a.dataset / "metrics.csv")
    ref = ref[(ref.extractor == "cnn1d") & (ref.cv_seed == cfg["cv_seed"])].set_index(["instance", "task"])
    ok, rows = True, []
    for s in a.seeds:
        inst = f"seed{s:03d}"
        theirs = a.phase1 / a.dataset / "features" / "cnn1d" / f"{inst}.npz"
        ours = a.out / a.dataset / "features" / "cnn" / "default" / f"{inst}.npz"
        m1 = metrics(theirs, cfg, labels, tasks)
        m2 = metrics(ours, cfg, labels, tasks) if ours.exists() else {}
        if ours.exists():
            A, B = np.load(theirs), np.load(ours)
            same_rows = np.array_equal(A["participant_id"], B["participant_id"]) and np.array_equal(A["cond"], B["cond"])
            rel = np.abs(A["feats"] - B["feats"]) / (np.abs(A["feats"]) + 1e-6)
            ok &= same_rows
        for t in tasks:
            r = ref.loc[(inst, t)]
            match1 = abs(m1[t][0] - r.auc) < 1e-5 and abs(m1[t][1] - r.balanced_accuracy) < 1e-5
            ok &= match1
            row = dict(seed=s, task=t, phase1_auc=r.auc, our_probe_on_phase1_feats=m1[t][0], probe_match=match1)
            if t in m2:
                row.update(our_features_auc=m2[t][0], same_rows=same_rows, feat_max_rel_diff=rel.max(),
                           metric_match=abs(m2[t][0] - r.auc) < 1e-3)
                ok &= row["metric_match"]
            rows.append(row)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.round(5).to_string(index=False))
    agg = df.groupby("task").agg(phase1=("phase1_auc", "mean"), phase1_sd=("phase1_auc", "std"),
                                 **({"ours": ("our_features_auc", "mean"), "ours_sd": ("our_features_auc", "std")}
                                    if "our_features_auc" in df else {}))
    print("\nmulti-seed AUC (cv_seed %d, %d seeds):" % (cfg["cv_seed"], len(a.seeds)))
    print(agg.round(4).to_string())
    print("\nSTEP 0:", "MATCH" if ok else "MISMATCH - stop and investigate")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
