#!/usr/bin/env python
"""Step (a''), step 0: the reference must reproduce step (a')'s combined + Hjorth features bit for
bit, and every other level must be sane on real data, before the main run.

    python optimization/multiscale_step0.py --dataset ds003478 --data-root $S/data --out $OUT \
        --step-ap $A/11-11-26 [--seeds 0-2] [--probe --jobs 8]

Reads <out>/multiscale/<level>/<dataset>/features/cnn/<BASE>/seed###.npz (extract.py --multiscale).
1. ref, seeds --seeds: features, dtype and block rows equal step (a')'s "hjorth" level of the
   combined architecture bit for bit; AUC and balanced accuracy equal its per_seed.csv.
2. every other level, seed 0: same block rows as ref, float32, finite, expected dimension, and
   the columns that must not change are bit-identical (same weights): ms2 ends with ref's
   columns, ms3 = ms2's columns then last3's, fb_ms2 ends with fb's columns.
3. --probe: probe seed 0 of every level (all tasks) and time it. Writes
   <out>/multiscale_step0/probe_<dataset>.csv for multiscale_estimate.py. NOT a result.
Exits 1 on any mismatch.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimization import multiscale  # noqa: E402
from optimization.common import dataset_config  # noqa: E402
from optimization.evaluate_dataset import eval_file, load  # noqa: E402
from optimization.prep_step0 import same_file, seed_range  # noqa: E402
from untrained_eeg.tasks import task_labels, task_names  # noqa: E402

COMBINED = "cov_channels=32,n_filters=256,readout=meanstd_ppv_max"  # step (a')'s folder name
DIMS = {"ref": 1536, "ms2": 2304, "last3": 3072, "ms3": 5376, "fb": 1536, "fb_ms2": 2304}
# (level, its columns, level they must equal, those columns): same weights, same numbers
SAME = [("ms2", slice(768, None), "ref", slice(None)),
        ("ms3", slice(0, 2304), "ms2", slice(None)),
        ("ms3", slice(2304, None), "last3", slice(None)),
        ("fb_ms2", slice(768, None), "fb", slice(None))]


def check_ref(a, cfg, labels, tasks, root):
    ok, rows = True, []
    theirs_dir = a.step_ap / "readout" / "hjorth" / a.dataset
    ps = theirs_dir / "per_seed.csv"
    if not ps.exists():
        print(f"missing step (a') results: {ps}")
        return False, rows
    ps = pd.read_csv(ps)
    ps = ps[ps.config == COMBINED]
    for s in a.seeds:
        ours = root("ref") / f"seed{s:03d}.npz"
        theirs = theirs_dir / "features" / "cnn" / COMBINED / ours.name
        if not ours.exists() or not theirs.exists():
            print(f"missing: {ours if not ours.exists() else theirs}")
            ok = False
            continue
        bits = same_file(ours, theirs)
        for r in eval_file(ours, cfg, labels, tasks, a.dataset):
            t = ps[(ps.seed == s) & (ps.task == r["task"])]
            match = len(t) > 0 and abs(t.auc.iloc[0] - r["auc"]) < 5e-6 \
                and abs(t.balanced_accuracy.iloc[0] - r["balanced_accuracy"]) < 5e-6
            rows.append(dict(seed=s, task=r["task"], bit_identical=bits, auc=r["auc"],
                             step_ap_auc=t.auc.iloc[0] if len(t) else np.nan, metrics_match=match))
            ok &= bits and match
    return ok, rows


def check_levels(a, root):
    ok, rows = True, []
    if not (root("ref") / "seed000.npz").exists():
        return False, rows
    ref = load(root("ref") / "seed000.npz")
    data = {"ref": ref}
    for lv in multiscale.LEVELS:
        if lv == "ref":
            continue
        f = root(lv) / "seed000.npz"
        if not f.exists():
            rows.append(dict(level=lv, ok=False, note="missing"))
            ok = False
            continue
        d = data[lv] = load(f)
        X, m = d["feats"], d["meta"]
        checks = dict(same_blocks=np.array_equal(d["pid"], ref["pid"]) and np.array_equal(d["cond"], ref["cond"]),
                      float32=X.dtype == np.float32, finite=bool(np.isfinite(X).all()),
                      dim=X.shape[1] == m["feature_dim"] == DIMS[lv])
        rows.append(dict(level=lv, checks=checks, n_features=X.shape[1], arch_run=m["arch_run"],
                         extract_s=round(m["extract_seconds"], 1), input_s=round(m["input_seconds"], 1)))
    for lv, cols, other, ocols in SAME:  # columns that the same weights must reproduce exactly
        if lv in data and other in data:
            same = np.array_equal(data[lv]["feats"][:, cols], data[other]["feats"][:, ocols])
            next(r for r in rows if r["level"] == lv)["checks"][f"cols_eq_{other}"] = same
    for r in rows:
        c = r.pop("checks", {})
        r["ok"] = r.get("ok", True) and all(c.values())
        r["note"] = r.get("note", " ".join(k for k, v in c.items() if not v))
        ok &= r["ok"]
    return ok, rows


def probe_all(a, cfg, labels, tasks, root):
    """Part 3: time the probe of seed 0 at every level (for the CPU-hour estimate)."""
    from joblib import Parallel, delayed
    dest = a.out / "multiscale_step0" / f"probe_{a.dataset}.csv"
    if dest.exists():
        raise FileExistsError(dest)
    files = [(lv, root(lv) / "seed000.npz") for lv in multiscale.LEVELS]
    t0 = time.time()
    res = Parallel(n_jobs=a.jobs)(delayed(eval_file)(f, cfg, labels, tasks, a.dataset) for _, f in files)
    rows = [dict(level=lv, **r) for (lv, _), rr in zip(files, res) for r in rr]
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest, index=False, float_format="%.6g")
    print(f"wrote {dest} ({len(rows)} rows, {time.time() - t0:.0f} s wall)")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--step-ap", type=Path, required=True, help="step (a') results, e.g. $A/11-11-26")
    ap.add_argument("--seeds", type=seed_range, default=[0, 1, 2])
    ap.add_argument("--probe", action="store_true", help="also time the probe of seed 0 at every level")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    cfg = dataset_config(a.dataset, a.data_root)
    labels, tasks = task_labels(cfg), task_names(cfg)
    root = lambda lv: a.out / "multiscale" / lv / a.dataset / "features" / "cnn" / multiscale.BASE  # noqa: E731
    pd.set_option("display.width", 200)

    ok1, rows = check_ref(a, cfg, labels, tasks, root)
    print(pd.DataFrame(rows).round(6).to_string(index=False))
    ok2, rows = check_levels(a, root)
    print("\n" + pd.DataFrame(rows).to_string(index=False))
    if a.probe and ok1 and ok2:
        p = probe_all(a, cfg, labels, tasks, root)
        t = p.groupby("level").agg(probe_s=("probe_seconds", "sum"), auc=("auc", "mean")).round(3)
        print("\nseed 0 only (timing check, not a result):\n" + t.to_string())
    ok = ok1 and ok2
    print(f"\nSTEP 0 (a'') {a.dataset}:", "MATCH" if ok else "MISMATCH - stop and investigate")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    # one BLAS thread, as in every earlier probe: the thread count changes summation order
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        main()
