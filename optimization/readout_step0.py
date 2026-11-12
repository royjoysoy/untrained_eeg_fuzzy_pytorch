#!/usr/bin/env python
"""Step (a'), step 0: the reference readout must reproduce step (a) exactly, and every other
readout level must be sane on real data, before the main run.

    python optimization/readout_step0.py --dataset ds003478 --data-root $S/data --out $OUT \
        --step-a $A/11-5-26 $A/11-5-26b [--seeds 0-2] [--probe --jobs 8]

Reads <out>/readout/<level>/<dataset>/features/cnn/<arch>/seed###.npz (extract.py --readout).
1. ref, seeds --seeds, every architecture: features, dtype and block rows equal the step-(a)
   file bit for bit; c1 AUC and balanced accuracy equal step (a)'s per_seed.csv.
2. every other level, seed 0: same block rows as ref; all values finite; SPD blocks positive
   definite (the tangent space can be fitted); levels that ADD to the readout (ref_logvar,
   ref_tcov32) keep the reference's own columns bit for bit.
3. --probe: probe seed 0 of every level (all tasks) and time it. Writes
   <out>/readout_step0/probe_<dataset>.csv for readout_estimate.py. One seed only: NOT a result.
Exits 1 on any mismatch.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimization import readout  # noqa: E402
from optimization.arch import from_name  # noqa: E402
from optimization.common import dataset_config  # noqa: E402
from optimization.evaluate import RiemannTangent, c1  # noqa: E402
from optimization.evaluate_dataset import eval_file, load  # noqa: E402
from optimization.prep_step0 import same_file, seed_range  # noqa: E402
from untrained_eeg.tasks import task_data, task_labels, task_names  # noqa: E402


def check_ref(a, cfg, labels, tasks, root):
    """Part 1, as prep_step0.py: bit-identical features and identical metrics."""
    ok, rows = True, []

    def find(rel):
        return next((d / a.dataset / "features" / rel for d in a.step_a
                     if (d / a.dataset / "features" / rel).exists()), None)

    step_a = pd.concat([pd.read_csv(f) for d in a.step_a for f in [d / a.dataset / "per_seed.csv",
                        d / a.dataset / "per_seed_search.csv"] if f.exists()], ignore_index=True)
    archs = sorted(p.name for p in (root("ref") / "cnn").glob("*"))
    if not archs:
        print("no reference features found")
        return False, rows
    for arch in archs:
        for s in a.seeds:
            ours = root("ref") / "cnn" / arch / f"seed{s:03d}.npz"
            theirs = find(Path("cnn") / arch / ours.name)
            if not ours.exists() or theirs is None:
                print(f"missing: {ours if not ours.exists() else 'step (a) ' + arch + ' ' + ours.name}")
                ok = False
                continue
            bits = same_file(ours, theirs)
            f = np.load(ours)
            for t in tasks:
                X, y, g, _ = task_data(f["feats"], f["cond"], f["participant_id"], t, labels)
                _, auc, bacc = c1(X, y, g, t, cfg)
                r = step_a[(step_a.config == arch) & (step_a.seed == s) & (step_a.task == t)]
                match = len(r) > 0 and abs(r.auc.iloc[0] - auc) < 5e-6 and abs(r.balanced_accuracy.iloc[0] - bacc) < 5e-6
                rows.append(dict(arch=arch, seed=s, task=t, bit_identical=bits, auc=auc,
                                 step_a_auc=r.auc.iloc[0] if len(r) else np.nan, metrics_match=match))
                ok &= bits and match
    return ok, rows


def check_levels(a, cfg, root):
    """Part 2: every level other than ref, seed 0."""
    ok, rows = True, []
    for lv in readout.LEVELS:
        if lv == "ref":
            continue
        if not readout.applies(lv, cfg):
            print(f"{lv:13s} does not apply to {a.dataset}")
            continue
        for arch_dir in sorted((root("ref") / "cnn").glob("*")):
            f = root(lv) / "cnn" / arch_dir.name / "seed000.npz"
            if not f.exists():
                rows.append(dict(level=lv, arch=arch_dir.name, ok=False, note="missing"))
                ok = False
                continue
            d, r = load(f), load(arch_dir / "seed000.npz")
            X, R, m = d["feats"], r["feats"], d["meta"]
            checks = dict(same_blocks=np.array_equal(d["pid"], r["pid"]) and np.array_equal(d["cond"], r["cond"]),
                          float32=X.dtype == np.float32, finite=bool(np.isfinite(X).all()),
                          dim=X.shape[1] == m["feature_dim"])
            min_eig = np.nan
            for s, n in m.get("spd_blocks", []):
                min_eig = float(np.linalg.eigvalsh(RiemannTangent._mats(X, s, n)).min())
                checks["spd"] = min_eig > 0
            if lv in ("ref_logvar", "ref_tcov32"):  # the reference's own columns, untouched
                k = from_name(arch_dir.name).cov_channels or 0  # ref's off-diagonal covariance columns
                n_cov = int(k) * (int(k) - 1) // 2
                head = R.shape[1] - n_cov
                same = np.array_equal(X[:, :head], R[:, :head])
                if lv == "ref_logvar" and n_cov:
                    same &= np.array_equal(X[:, -n_cov:], R[:, -n_cov:])
                checks["ref_columns"] = bool(same)
            good = all(checks.values())
            rows.append(dict(level=lv, arch=arch_dir.name, ok=good, n_features=X.shape[1], min_eig=min_eig,
                             extract_s=round(m["extract_seconds"], 1),
                             note=" ".join(k for k, v in checks.items() if not v)))
            ok &= good
    return ok, rows


def probe_all(a, cfg, labels, tasks, root):
    """Part 3: time the probe of seed 0 at every level (for the CPU-hour estimate)."""
    from joblib import Parallel, delayed
    dest = a.out / "readout_step0" / f"probe_{a.dataset}.csv"
    if dest.exists():
        raise FileExistsError(dest)
    files = [(lv, f) for lv in readout.LEVELS if readout.applies(lv, cfg)
             for f in sorted((root(lv) / "cnn").glob("*/seed000.npz"))]
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
    ap.add_argument("--step-a", type=Path, nargs="+", required=True)
    ap.add_argument("--seeds", type=seed_range, default=[0, 1, 2])
    ap.add_argument("--probe", action="store_true", help="also time the probe of seed 0 at every level")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    cfg = dataset_config(a.dataset, a.data_root)
    labels, tasks = task_labels(cfg), task_names(cfg)
    root = lambda lv: a.out / "readout" / lv / a.dataset / "features"  # noqa: E731
    pd.set_option("display.width", 200)

    ok1, rows = check_ref(a, cfg, labels, tasks, root)
    print(pd.DataFrame(rows).round(6).to_string(index=False))
    ok2, rows = check_levels(a, cfg, root)
    print("\n" + pd.DataFrame(rows).to_string(index=False))
    if a.probe and ok1 and ok2:
        p = probe_all(a, cfg, labels, tasks, root)
        t = p.groupby(["level", "config"]).agg(probe_s=("probe_seconds", "sum"), auc=("auc", "mean")).round(3)
        print("\nseed 0 only (timing check, not a result):\n" + t.to_string())
    ok = ok1 and ok2
    print(f"\nSTEP 0 (a') {a.dataset}:", "MATCH" if ok else "MISMATCH - stop and investigate")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    # one BLAS thread, as in step (a)'s probes: the thread count changes summation order
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        main()
