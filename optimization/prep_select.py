#!/usr/bin/env python
"""Step (b): choose the preprocessing level without peeking, and summarize.

    python optimization/prep_select.py nested --dataset ds003478 --data-root $S/data --out $OUT --jobs 16
    python optimization/prep_select.py summary --out $OUT

Inputs: <out>/prep/<level>/<dataset>/{features/, per_seed.csv, minirocket.csv} from
extract.py, evaluate_dataset.py and minirocket.py run with --prep <level>.
A level that does not apply to a dataset (prep.applies) IS the reference there.

nested  -> <out>/prep_summary/nested_<dataset>.csv, selection inside outer training folds:
  cnn:<arch>          per weight seed: best level among the applicable ones (incl. ref)
  handcrafted         best (level, family) pair, families incl. all_combined
  handcrafted@<level> best family at that fixed level (the level's handcrafted score)
summary -> <out>/prep_summary/:
  results.csv         dataset x task x kind x level: AUC and balanced accuracy mean / SD /
                      min / max over seeds, delta AUC vs the reference of the same kind
                      (DESCRIPTIVE: every level is scored on the same folds)
  factor_effects.csv  per kind and level: mean delta AUC over the tasks where it applies
  lodo.csv            leave-one-dataset-out: level with the best mean rank on the OTHER
                      datasets' tasks (a level that does not apply there = the reference)
  nested.csv          nested selection, mean / SD / min / max over seeds
  headline.csv        per task: reference, nested and LODO level, for every kind
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
from optimization import prep  # noqa: E402
from optimization.common import DATASETS, dataset_config  # noqa: E402
from optimization.evaluate import nested  # noqa: E402
from optimization.evaluate_dataset import combined_handcrafted, done_files, load  # noqa: E402
from untrained_eeg.tasks import task_data, task_labels, task_names  # noqa: E402

AGG = dict(n_seeds=("seed", lambda s: int(s.notna().sum()) or 1),
           auc_mean=("auc", "mean"), auc_sd=("auc", "std"), auc_min=("auc", "min"), auc_max=("auc", "max"),
           bacc_mean=("balanced_accuracy", "mean"), bacc_sd=("balanced_accuracy", "std"),
           bacc_min=("balanced_accuracy", "min"), bacc_max=("balanced_accuracy", "max"))


def levels_of(dataset):
    cfg = dataset_config(dataset, None)
    return [lv for lv in prep.LEVELS if prep.applies(lv, cfg)]


# --------------------------------------------------------------------------- nested
def run_nested(a):
    cfg = dataset_config(a.dataset, a.data_root)
    labels, tasks = task_labels(cfg), task_names(cfg)
    dest = a.out / "prep_summary" / f"nested_{a.dataset}.csv"
    if dest.exists():
        raise FileExistsError(dest)
    root = lambda lv: a.out / "prep" / lv / a.dataset / "features"
    levels = levels_of(a.dataset)
    ref_rows = None

    def same_rows(f):  # every level must describe the same blocks as the reference
        nonlocal ref_rows
        with np.load(f) as z:
            key = (tuple(z["participant_id"]), tuple(z["cond"]))
        ref_rows = ref_rows or key
        if key != ref_rows:
            raise ValueError(f"block rows differ from the reference: {f}")

    def cands_of(files, task):
        out = {}
        for name, f in files.items():
            d = f if isinstance(f, dict) else load(f)
            X, y, g, _ = task_data(d["feats"], d["cond"], d["pid"], task, labels)
            out[name] = (X, d["meta"].get("spd_blocks", ()))
        return out, y, g

    def nest(task, kind, files, seed=np.nan):
        t0 = time.time()
        cands, y, g = cands_of(files, task)
        _, auc, bacc, chosen = nested(cands, y, g, task, cfg)
        return dict(dataset=a.dataset, task=task, kind=kind, seed=seed, n_candidates=len(cands),
                    auc=auc, balanced_accuracy=bacc, nested_seconds=time.time() - t0,
                    chosen=json.dumps(pd.Series(chosen).value_counts().to_dict()))

    jobs = []
    archs = sorted({p.parent.name for p in (root("ref") / "cnn").glob("*/seed*.npz")})
    for arch in archs:
        files = {lv: done_files((root(lv) / "cnn" / arch).glob("seed*.npz")) for lv in levels}
        if [lv for lv, fs in files.items() if not fs]:
            raise SystemExit(f"{arch}: no features at {[lv for lv, fs in files.items() if not fs]}")
        seeds = sorted(int(f.stem[4:]) for f in files["ref"])
        for lv, fs in files.items():
            if {int(f.stem[4:]) for f in fs} != set(seeds):
                raise ValueError(f"{arch} at {lv}: seeds differ from the reference")
            for f in fs[:1]:
                same_rows(f)
        for s in seeds:
            per = {lv: root(lv) / "cnn" / arch / f"seed{s:03d}.npz" for lv in levels}
            jobs += [(t, f"cnn:{arch}", per, s) for t in tasks]
    hand = {}  # (level, family) -> file or combined dict
    for lv in [lv for lv in levels if prep.for_handcrafted(lv)]:
        fs = done_files((root(lv) / "handcrafted").glob("*.npz"))
        if not fs:
            continue
        same_rows(fs[0])
        fam = {f.stem: f for f in fs}
        fam["all_combined"] = combined_handcrafted(fs)
        jobs += [(t, f"handcrafted@{lv}", fam) for t in tasks]
        hand.update({f"{lv}|{k}": v for k, v in fam.items()})
    if hand:
        jobs += [(t, "handcrafted", hand) for t in tasks]
    rows = Parallel(n_jobs=a.jobs)(delayed(nest)(*j) for j in jobs)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest, index=False, float_format="%.6g")
    print(f"wrote {dest} ({len(rows)} rows)")


# --------------------------------------------------------------------------- summary
def per_seed_table(out):
    """All per-seed rows, with `kind` = cnn:<arch>, handcrafted:<family> or minirocket."""
    parts = []
    for f in sorted(out.glob("prep/*/*/per_seed.csv")) + sorted(out.glob("prep/*/*/minirocket.csv")):
        d = pd.read_csv(f)
        d["level"] = f.parts[-3]
        d["kind"] = np.where(d.kind == "cnn", "cnn:" + d.config,
                             np.where(d.kind == "handcrafted", "handcrafted:" + d.config, d.kind))
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def stand_in(level, have):
    """What a level means where it was not run: the same input as sf256 (kernel x2 only changes
    the CNN), otherwise the reference (the level does not apply to that dataset or kind)."""
    if level in have:
        return level
    return "sf256" if prep.LEVELS[level].kernel_x2 and "sf256" in have else "ref"


def fill_levels(means):
    """Every level for every dataset x task x kind, missing ones filled by their stand-in."""
    rows = []
    for _, g in means.groupby(["dataset", "task", "kind"]):
        have = set(g.level)
        for lv in prep.LEVELS:
            src = g[g.level == stand_in(lv, have)]
            rows.append(src.assign(level=lv, applies=lv in have))
    return pd.concat(rows, ignore_index=True)


def stats(g):
    return {k: g[c].agg(fn) for k, (c, fn) in AGG.items()}


def lodo(per_seed):
    """Per kind and held-out dataset: the level with the best mean rank on the other datasets."""
    means = per_seed.groupby(["dataset", "task", "kind", "level"]).auc.mean().reset_index()
    full = fill_levels(means)
    full["rank"] = full.groupby(["dataset", "task", "kind"]).auc.rank(ascending=False)
    rows = []
    for kind, k in full.groupby("kind"):
        for held in sorted(k.dataset.unique()):
            other = k[k.dataset != held]
            if other.empty:
                continue
            best = other.groupby("level")["rank"].mean().idxmin()
            d = per_seed[(per_seed.kind == kind) & (per_seed.dataset == held)]
            used = stand_in(best, set(d.level))
            for task, g in d[d.level == used].groupby("task"):
                rows.append(dict(dataset=held, task=task, kind=kind, chosen=best, used=used, **stats(g)))
    return pd.DataFrame(rows)


def missing_levels(ps):
    """Applicable levels without results, per dataset and kind (they must not pass as the reference)."""
    out = []
    for (ds, kind), g in ps.groupby(["dataset", "kind"]):
        cfg = dataset_config(ds, None)
        want = [lv for lv in prep.LEVELS if prep.applies(lv, cfg)]
        if kind.startswith("handcrafted:"):
            want = [lv for lv in want if prep.for_handcrafted(lv)]
        elif kind == "minirocket":
            want = [lv for lv in want if prep.for_minirocket(lv)]
        out += [f"{ds} {kind} {lv}" for lv in want if lv not in set(g.level)]
    return out


def run_summary(a):
    dest = a.out / "prep_summary"
    for f in ("results.csv", "factor_effects.csv", "lodo.csv", "nested.csv", "headline.csv"):
        if (dest / f).exists():
            raise FileExistsError(dest / f)
    dest.mkdir(parents=True, exist_ok=True)
    ps = per_seed_table(a.out)
    missing = missing_levels(ps)
    if missing and not a.allow_missing:
        raise SystemExit("levels that apply but have no results (failed jobs?):\n  " + "\n  ".join(missing)
                         + "\nrerun them, or pass --allow-missing for a partial look")
    nest = pd.concat([pd.read_csv(f) for f in sorted(dest.glob("nested_*.csv"))], ignore_index=True) \
        if list(dest.glob("nested_*.csv")) else pd.DataFrame()

    # handcrafted per level = nested choice of family at that level (one value, no seeds)
    if len(nest):
        h = nest[nest.kind.str.startswith("handcrafted@")]
        h = h.assign(level=h.kind.str.removeprefix("handcrafted@"), kind="handcrafted")
        ps = pd.concat([ps, h[["dataset", "task", "kind", "level", "seed", "auc", "balanced_accuracy"]]],
                       ignore_index=True)

    res = ps.groupby(["dataset", "task", "kind", "level"], sort=False).agg(**AGG).reset_index()
    res["factor"] = res.level.map(lambda lv: prep.LEVELS[lv].factor)
    ref = res[res.level == "ref"].set_index(["dataset", "task", "kind"]).auc_mean
    res["delta_auc_vs_ref"] = [r.auc_mean - ref.get((r.dataset, r.task, r.kind), np.nan) for r in res.itertuples()]
    res.to_csv(dest / "results.csv", index=False, float_format="%.4g")

    eff = res[res.level != "ref"].groupby(["kind", "factor", "level"]).agg(
        n_tasks=("task", "size"), delta_auc_mean=("delta_auc_vs_ref", "mean"),
        delta_auc_min=("delta_auc_vs_ref", "min"), delta_auc_max=("delta_auc_vs_ref", "max")).reset_index()
    eff.to_csv(dest / "factor_effects.csv", index=False, float_format="%.4g")

    lo = lodo(ps[~ps.kind.str.startswith("handcrafted:")])
    lo.to_csv(dest / "lodo.csv", index=False, float_format="%.4g")

    nsum = pd.DataFrame()
    if len(nest):
        nsum = nest[~nest.kind.str.startswith("handcrafted@")].groupby(["dataset", "task", "kind"]).agg(**AGG).reset_index()
        nsum.to_csv(dest / "nested.csv", index=False, float_format="%.4g")

    head = []
    for (ds, task), g in res.groupby(["dataset", "task"]):
        row = dict(dataset=ds, task=task)
        for kind in sorted(k for k in g.kind.unique() if not k.startswith("handcrafted:")):
            r = g[(g.kind == kind) & (g.level == "ref")]
            row[f"{kind}|ref"] = r.auc_mean.iloc[0] if len(r) else np.nan
            n = nsum[(nsum.dataset == ds) & (nsum.task == task) & (nsum.kind == kind)] if len(nsum) else nsum
            row[f"{kind}|nested"] = n.auc_mean.iloc[0] if len(n) else np.nan
            lsel = lo[(lo.dataset == ds) & (lo.task == task) & (lo.kind == kind)] if len(lo) else lo
            row[f"{kind}|lodo"] = lsel.auc_mean.iloc[0] if len(lsel) else np.nan
            row[f"{kind}|lodo_level"] = lsel.chosen.iloc[0] if len(lsel) else ""
        head.append(row)
    head = pd.DataFrame(head)
    head.to_csv(dest / "headline.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    print(eff.round(4).to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["nested", "summary"])
    p.add_argument("--dataset", choices=list(DATASETS))
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--allow-missing", action="store_true", help="summary: allow levels without results")
    a = p.parse_args()
    from optimization.runinfo import run_info
    with run_info(a.out, f"prep_{a.what}", a.dataset):
        run_nested(a) if a.what == "nested" else run_summary(a)


if __name__ == "__main__":
    main()
