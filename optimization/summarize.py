#!/usr/bin/env python
"""Combine all datasets: Step 3 tables, leave-one-dataset-out selection, phase-1 references.

    python optimization/summarize.py --out results/optimization/11-5-26 \
        --phase1 /project/rrg-glatard/shared/untrained_eeg/results/phase1

Reads <out>/<dataset>/per_seed.csv and nested.csv. Writes in <out>/summary/:
  results.csv     one row per dataset x task x config: n_seeds, AUC and balanced accuracy
                  mean / SD / min / max over weight seeds, delta AUC vs the current cnn1d,
                  feature_dim, receptive field (s), runtime. DESCRIPTIVE: every config is
                  scored on the same folds, so its maximum is not an unbiased estimate.
  lodo.csv        (b) leave-one-dataset-out: for each held-out dataset, the config (and the
                  handcrafted family) with the best mean rank on the OTHER datasets' tasks,
                  then its per-seed results on the held-out dataset's tasks.
  nested.csv      (a) per-task nested selection, mean / SD / min / max over seeds.
  references.csv  phase-1 extractors on the same tasks (same probe, folds and instances):
                  cnn1d, shallow, eegnet, scattering, best foundation model.
  headline.csv    per task: current cnn1d, nested and LODO untrained CNN, best handcrafted
                  (nested and LODO), MiniRocket (mean over its seeds), phase-1 references.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

AGG = dict(n_seeds=("seed", lambda s: int(s.notna().sum()) or 1),
           auc_mean=("auc", "mean"), auc_sd=("auc", "std"), auc_min=("auc", "min"), auc_max=("auc", "max"),
           bacc_mean=("balanced_accuracy", "mean"), bacc_sd=("balanced_accuracy", "std"),
           bacc_min=("balanced_accuracy", "min"), bacc_max=("balanced_accuracy", "max"))


def lodo(per_seed, kind):
    """Pick one config per held-out dataset from the other datasets' tasks (mean rank of mean AUC)."""
    d = per_seed[per_seed.kind == kind]
    if kind == "cnn":
        d = d[d.factor != "manual"]
    means = d.groupby(["dataset", "task", "config"]).auc.mean().reset_index()
    means["rank"] = means.groupby(["dataset", "task"]).auc.rank(ascending=False)
    rows = []
    for held in sorted(d.dataset.unique()):
        other = means[means.dataset != held]
        if other.empty:
            continue
        best = other.groupby("config")["rank"].mean().idxmin()
        sel = d[(d.dataset == held) & (d.config == best)]
        for task, g in sel.groupby("task"):
            rows.append(dict(dataset=held, task=task, kind=f"{kind}_lodo", chosen=best,
                             n_seeds=int(g.seed.notna().sum()) or 1,
                             auc_mean=g.auc.mean(), auc_sd=g.auc.std(), auc_min=g.auc.min(), auc_max=g.auc.max(),
                             bacc_mean=g.balanced_accuracy.mean(), bacc_sd=g.balanced_accuracy.std()))
    return pd.DataFrame(rows)


def references(phase1: Path, datasets):
    rows = []
    for ds in datasets:
        f = phase1 / ds / "metrics.csv"
        if not f.exists():
            continue
        m = pd.read_csv(f)
        m = m[m.cv_seed == 0]
        for (task, ext, grp), g in m.groupby(["task", "extractor", "group"]):
            rows.append(dict(dataset=ds, task=task, extractor=ext, group=grp, n=len(g),
                             auc_mean=g.auc.mean(), auc_sd=g.auc.std(), bacc_mean=g.balanced_accuracy.mean()))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--phase1", type=Path, default=None)
    a = p.parse_args()
    dest = a.out / "summary"
    if dest.exists():
        raise FileExistsError(dest)
    files = sorted(a.out.glob("*/per_seed.csv")) + sorted(a.out.glob("*/minirocket.csv"))
    per_seed = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    nfiles = sorted(a.out.glob("*/nested.csv"))
    dest.mkdir(parents=True)

    keys = ["dataset", "task", "kind", "factor", "value", "config"]
    res = per_seed.groupby(keys, dropna=False, sort=False).agg(
        **AGG, feature_dim=("feature_dim", "first"), receptive_field_s=("receptive_field_s", "first"),
        runtime_s_per_seed=("extract_seconds", "mean")).reset_index()
    base = res[(res.kind == "cnn") & (res.config == "default")].set_index(["dataset", "task"]).auc_mean
    res["delta_auc_vs_current"] = res.apply(lambda r: r.auc_mean - base.get((r.dataset, r.task), np.nan), axis=1)
    res.to_csv(dest / "results.csv", index=False, float_format="%.4g")

    lo = pd.concat([lodo(per_seed, "cnn"), lodo(per_seed, "handcrafted")], ignore_index=True)
    lo.to_csv(dest / "lodo.csv", index=False, float_format="%.4g")

    nest = pd.DataFrame()
    if nfiles:
        n = pd.concat([pd.read_csv(f) for f in nfiles], ignore_index=True)
        nest = n.groupby(["dataset", "task", "kind"]).agg(**AGG).reset_index()
        nest.to_csv(dest / "nested.csv", index=False, float_format="%.4g")

    ref = references(a.phase1, per_seed.dataset.unique()) if a.phase1 else pd.DataFrame()
    if not ref.empty:
        ref.to_csv(dest / "references.csv", index=False, float_format="%.4g")

    # headline: one row per task
    head = []
    hand = res[res.kind == "handcrafted"]
    for (ds, task), g in res[res.kind == "cnn"].groupby(["dataset", "task"]):
        row = dict(dataset=ds, task=task)
        cur = g[g.config == "default"]
        row["cnn1d_current"] = cur.auc_mean.iloc[0] if len(cur) else np.nan
        row["cnn_best_descriptive"] = g.auc_mean.max()
        row["cnn_best_descriptive_config"] = g.loc[g.auc_mean.idxmax(), "config"]
        for kind, col in (("cnn_grid", "cnn_nested"), ("cnn_search", "cnn_search_nested"),
                          ("handcrafted", "handcrafted_nested")):
            sel = nest[(nest.dataset == ds) & (nest.task == task) & (nest.kind == kind)] if len(nest) else nest
            row[col] = sel.auc_mean.iloc[0] if len(sel) else np.nan
        for kind in ("cnn_lodo", "handcrafted_lodo"):
            sel = lo[(lo.dataset == ds) & (lo.task == task) & (lo.kind == kind)] if len(lo) else lo
            row[kind] = sel.auc_mean.iloc[0] if len(sel) else np.nan
            row[kind + "_config"] = sel.chosen.iloc[0] if len(sel) else ""
        mr = res[(res.kind == "minirocket") & (res.dataset == ds) & (res.task == task)]
        row["minirocket"] = mr.auc_mean.iloc[0] if len(mr) else np.nan
        h = hand[(hand.dataset == ds) & (hand.task == task)]
        if len(h):
            row["handcrafted_best_descriptive"] = h.auc_mean.max()
            row["handcrafted_best_family"] = h.loc[h.auc_mean.idxmax(), "config"]
        if not ref.empty:
            r = ref[(ref.dataset == ds) & (ref.task == task)].set_index("extractor")
            for e in ("cnn1d", "shallow", "eegnet", "scattering"):
                row[f"phase1_{e}"] = r.auc_mean.get(e, np.nan)
            fm = r[r.group == "foundation"]
            if len(fm):
                row["phase1_best_foundation"] = fm.auc_mean.max()
                row["phase1_best_foundation_model"] = fm.auc_mean.idxmax()
        head.append(row)
    pd.DataFrame(head).to_csv(dest / "headline.csv", index=False, float_format="%.4g")
    print(pd.DataFrame(head).round(3).to_string())


if __name__ == "__main__":
    main()
