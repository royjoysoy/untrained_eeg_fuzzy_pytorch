#!/usr/bin/env python
"""Probe every extracted feature set of ONE dataset on all its tasks.

    python optimization/evaluate_dataset.py --dataset ds003478 --data-root $S/data \
        --out results/optimization/11-5-26 [--nested] [--jobs 16]

Writes in <out>/<dataset>/ (never overwrites):
  per_seed.csv  dataset, task, kind, factor, value, config, seed, n, auc, balanced_accuracy,
                feature_dim, receptive_field_s, extract_seconds. CNN rows use the phase-1 c1
                probe (untrained_eeg.tasks.probe); handcrafted rows use the same splits and
                probe, with the Riemannian tangent space fitted inside training folds.
                `all_combined` = every handcrafted family concatenated.
  nested.csv    (--nested) selection inside the outer training folds:
                  cnn_grid     one row per seed: best ablation config chosen per outer fold
                  handcrafted  best family (or all combined) chosen per outer fold
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimization.common import dataset_config  # noqa: E402
from optimization.arch import ablation_grid  # noqa: E402
from optimization.evaluate import c1, generic, nested  # noqa: E402
from optimization.runinfo import run_info  # noqa: E402
from untrained_eeg.tasks import task_data, task_labels, task_names  # noqa: E402


def load(path):
    """Features exactly as phase1_probe.py reads them: float32, as stored (no cast to
    float64, which moves probabilities by ~1e-5 and can swap near-tied AUC pairs)."""
    with np.load(path, allow_pickle=False) as f:
        return dict(feats=f["feats"], cond=f["cond"], pid=f["participant_id"],
                    meta=json.loads(str(f["meta"])))


def eval_file(path, cfg, labels, tasks, dataset):
    d = load(path)
    m = d["meta"]
    rows = []
    for task in tasks:
        t0 = time.time()
        X, y, groups, _ = task_data(d["feats"], d["cond"], d["pid"], task, labels)
        if m.get("spd_blocks"):
            _, auc, bacc = generic(X, y, groups, task, cfg, m["spd_blocks"])
        else:
            _, auc, bacc = c1(X, y, groups, task, cfg)
        rows.append(dict(dataset=dataset, task=task, kind=m["kind"], factor=m.get("factor", "handcrafted"),
                         value=m.get("value", m["config"]), config=m["config"], seed=m.get("seed", np.nan),
                         n=len(y), auc=auc, balanced_accuracy=bacc, feature_dim=m["feature_dim"],
                         receptive_field_s=m.get("receptive_field_s", np.nan),
                         extract_seconds=m.get("extract_seconds", m.get("extract_seconds_all_families", np.nan)),
                         n_windows=m.get("n_windows", np.nan), n_params=m.get("n_params", np.nan),
                         probe_seconds=time.time() - t0))
    return rows


def combined_handcrafted(files):
    """All families side by side (same block rows), Riemannian column blocks shifted."""
    parts = [load(f) for f in files]
    spd, off = [], 0
    for p in parts:
        if not np.array_equal(p["pid"], parts[0]["pid"]) or not np.array_equal(p["cond"], parts[0]["cond"]):
            raise ValueError("handcrafted families have different block rows")
        spd += [[off + s, n] for s, n in p["meta"].get("spd_blocks", [])]
        off += p["feats"].shape[1]
    feats = np.hstack([p["feats"] for p in parts])
    return dict(feats=feats, cond=parts[0]["cond"], pid=parts[0]["pid"],
                meta=dict(kind="handcrafted", config="all_combined", feature_dim=feats.shape[1], spd_blocks=spd))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--nested", action="store_true")
    p.add_argument("--nested-seeds", type=int, default=None, help="limit nested CNN selection to the first N seeds")
    p.add_argument("--search-list", type=Path, default=None,
                   help="config names from random_search.py: adds nested 'cnn_search' rows")
    p.add_argument("--jobs", type=int, default=1)
    a = p.parse_args()
    root = a.out / a.dataset
    for f in ("per_seed.csv", "nested.csv"):
        if (root / f).exists():
            raise FileExistsError(root / f)
    with run_info(a.out, "evaluate", a.dataset, extra=dict(nested=a.nested)):
        _main(a, root)


def _main(a, root):
    cfg = dataset_config(a.dataset, a.data_root)
    labels, tasks = task_labels(cfg), task_names(cfg)

    cnn_files = sorted((root / "features" / "cnn").glob("*/seed*.npz"))
    hand_files = sorted((root / "features" / "handcrafted").glob("*.npz"))
    # rule 5: every feature set must describe the same blocks (same participants, same order)
    ref = None
    for f in cnn_files + hand_files:
        with np.load(f) as z:
            key = (tuple(z["participant_id"]), tuple(z["cond"]))
        ref = ref or key
        if key != ref:
            raise ValueError(f"block rows differ: {f}")

    rows = Parallel(n_jobs=a.jobs)(delayed(eval_file)(f, cfg, labels, tasks, a.dataset)
                                   for f in cnn_files + hand_files)
    rows = [r for rr in rows for r in rr]
    if hand_files:
        comb = combined_handcrafted(hand_files)
        tmp = root / "features" / "_combined.tmp.npz"
        np.savez(tmp, feats=comb["feats"], cond=comb["cond"], participant_id=comb["pid"],
                 meta=json.dumps(comb["meta"]))
        rows += eval_file(tmp, cfg, labels, tasks, a.dataset)
        tmp.unlink()
    pd.DataFrame(rows).to_csv(root / "per_seed.csv", index=False, float_format="%.6g")
    print(f"wrote {root / 'per_seed.csv'} ({len(rows)} rows)")

    if not a.nested:
        return
    nrows = []
    grid_names = {c.name() for _, _, c in ablation_grid()}
    search = set(a.search_list.read_text().split()) if a.search_list else set()
    by_seed = {}
    for f in cnn_files:
        by_seed.setdefault(int(f.stem[4:]), []).append(f)
    seeds = sorted(by_seed)[: a.nested_seeds]
    pick = lambda files, names: [f for f in files if f.parent.name in names]

    def nest(task, files, kind, seed, extra=None):
        t0 = time.time()
        cands = {}
        for f in files:
            d = load(f)
            X, y, groups, _ = task_data(d["feats"], d["cond"], d["pid"], task, labels)
            cands[d["meta"]["config"]] = (X, d["meta"].get("spd_blocks", ()))
        if extra:
            X, y, groups, _ = task_data(extra["feats"], extra["cond"], extra["pid"], task, labels)
            cands["all_combined"] = (X, extra["meta"]["spd_blocks"])
        _, auc, bacc, chosen = nested(cands, y, groups, task, cfg)
        return dict(dataset=a.dataset, task=task, kind=kind, seed=seed, n_candidates=len(cands),
                    auc=auc, balanced_accuracy=bacc, nested_seconds=time.time() - t0,
                    chosen=json.dumps(pd.Series(chosen).value_counts().to_dict()))

    jobs = [(t, pick(by_seed[s], grid_names), "cnn_grid", s) for t in tasks for s in seeds]
    if search:
        for s in seeds:
            if {f.parent.name for f in pick(by_seed[s], search)} != search:
                raise ValueError(f"seed {s}: search configs missing")
        jobs += [(t, pick(by_seed[s], search), "cnn_search", s) for t in tasks for s in seeds]
    nrows = Parallel(n_jobs=a.jobs)(delayed(nest)(*j) for j in jobs)
    if hand_files:
        nrows += [nest(t, hand_files, "handcrafted", np.nan, comb) for t in tasks]
    pd.DataFrame(nrows).to_csv(root / "nested.csv", index=False, float_format="%.6g")
    print(f"wrote {root / 'nested.csv'} ({len(nrows)} rows)")


if __name__ == "__main__":
    main()
