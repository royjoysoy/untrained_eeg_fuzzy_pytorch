#!/usr/bin/env python
"""CPU-hour estimate for step (a'') from measured times (run after multiscale step 0).

    python optimization/multiscale_estimate.py --out $OUT --step-ap $A/11-11-26 [--seeds 30] [--safety 1.5]

Measured inputs:
  extraction  step 0's seed-0 files: meta extract_seconds + input_seconds per (dataset, level),
              times --seeds (ref included: the main run extracts every level for all seeds).
  probes      step (a')'s combined + Hjorth probes (readout/hjorth/<dataset>/per_seed.csv,
              probe_seconds per seed) times the probe-cost ratio level / ref measured in step 0
              (multiscale_step0/probe_<dataset>.csv). A dataset probed in step 0 uses its own
              ratios; the others use the LARGEST ratio measured on any dataset (conservative).
  nested      step (a')'s nested selection of the combined architecture (10 levels), scaled
              to 6 levels and by the mean probe ratio.
Everything is multiplied by --safety. Prints CPU-hours per dataset and stage.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimization import multiscale, readout  # noqa: E402
from optimization.common import DATASETS, dataset_config  # noqa: E402

COMBINED = "cov_channels=32,n_filters=256,readout=meanstd_ppv_max"


def extract_seconds(out, ds):
    """{level: seconds of one seed} from step 0's files."""
    t = {}
    for lv in multiscale.LEVELS:
        f = out / "multiscale" / lv / ds / "features" / "cnn" / multiscale.BASE / "seed000.npz"
        if f.exists():
            with np.load(f) as z:
                m = json.loads(str(z["meta"]))
            t[lv] = m["extract_seconds"] + m.get("input_seconds", 0.0)
    return t


def probe_ratios(out):
    """{dataset: {level: probe seconds at the level / at ref}} for the datasets probed in step 0."""
    r = {}
    for f in sorted((out / "multiscale_step0").glob("probe_*.csv")):
        d = pd.read_csv(f).groupby(["dataset", "level"]).probe_seconds.sum()
        for ds in d.index.get_level_values(0).unique():
            r[ds] = (d[ds] / d[ds]["ref"]).to_dict()
    if not r:
        raise SystemExit("no multiscale_step0/probe_*.csv: run step 0 with --probe first")
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="step (a'') results (with step 0 done)")
    ap.add_argument("--step-ap", type=Path, required=True, help="step (a') results, e.g. $A/11-11-26")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--safety", type=float, default=1.5)
    a = ap.parse_args()
    ratios = probe_ratios(a.out)
    worst = {lv: max(r.get(lv, np.nan) for r in ratios.values()) for lv in multiscale.LEVELS}
    print("probe-cost ratios level / ref (seed 0):")
    print(pd.DataFrame(ratios).round(2).to_string(), "\n")
    rows = []
    for ds in DATASETS:
        levels = [lv for lv in multiscale.LEVELS if multiscale.applies(lv, dataset_config(ds, None))]
        ext = extract_seconds(a.out, ds)
        ps = a.step_ap / "readout" / "hjorth" / ds / "per_seed.csv"
        if not ext or not ps.exists():
            print(f"{ds}: step-0 features or step-(a') probes missing, skipped")
            continue
        ps = pd.read_csv(ps)
        ref_probe = ps[ps.config == COMBINED].groupby("seed").probe_seconds.sum().mean()
        ratio = ratios.get(ds, worst)
        miss = [lv for lv in levels if lv not in ext]
        if miss:
            print(f"{ds}: no step-0 features for {miss}")
        extract = sum(ext.get(lv, np.nan) for lv in levels) * a.seeds / 3600
        probes = sum(ref_probe * ratio[lv] for lv in levels) * a.seeds / 3600
        nf = a.step_ap / "readout_summary" / f"nested_{ds}.csv"
        nested = np.nan
        if nf.exists():
            n = pd.read_csv(nf)
            n_old = len([lv for lv in readout.LEVELS if readout.applies(lv, dataset_config(ds, None))])
            mean_ratio = np.mean([ratio[lv] for lv in levels])
            nested = (n[n.kind == f"cnn:{COMBINED}"].nested_seconds.sum() / 3600
                      * len(levels) / n_old * mean_ratio)
        rows.append(dict(dataset=ds, levels=len(levels), extract=extract, probes=probes, nested=nested,
                         ratios="own" if ds in ratios else "largest"))
    df = pd.DataFrame(rows).set_index("dataset")
    df.loc["TOTAL", ["levels", "extract", "probes", "nested"]] = df[["levels", "extract", "probes", "nested"]].sum()
    print(df.round(1).to_string())
    total = df.loc["TOTAL", ["extract", "probes", "nested"]].sum()
    print(f"\nmeasured-based total {total:.0f} CPU-h; with safety x{a.safety}: {total * a.safety:.0f} CPU-h "
          "(CPU-hours used; allocation is a little higher: step (a') used 80.7 of 87.8 allocated)")


if __name__ == "__main__":
    main()
