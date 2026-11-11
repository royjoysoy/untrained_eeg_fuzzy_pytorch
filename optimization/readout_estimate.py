#!/usr/bin/env python
"""CPU-hour estimate for step (a') from measured times (run after readout step 0).

    python optimization/readout_estimate.py --out $OUT --step-b $A/11-9-26 [--seeds 30] [--safety 1.5]

Measured inputs:
  extraction  step 0's seed-0 files: meta extract_seconds per (dataset, level, architecture);
              the main run extracts --seeds seeds at every level (ref included).
  probes      step (b)'s reference probes (prep/ref/<dataset>/per_seed.csv, probe_seconds per
              file and architecture), times the probe-cost ratio level / ref measured in step 0
              (readout_step0/probe_<dataset>.csv; median over the datasets probed there).
  nested      step (b)'s nested selection of each architecture (prep_summary/nested_<dataset>.csv),
              scaled by the number of candidate levels and by the mean probe ratio.
Everything is multiplied by --safety. Prints CPU-hours per dataset and stage.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimization import prep, readout  # noqa: E402
from optimization.common import DATASETS, dataset_config  # noqa: E402


def extract_seconds(out, ds):
    """{(level, arch): seconds of one seed} from step 0's files."""
    t = {}
    for f in (out / "readout").glob(f"*/{ds}/features/cnn/*/seed000.npz"):
        with np.load(f) as z:
            t[(f.parts[-6], f.parent.name)] = json.loads(str(z["meta"]))["extract_seconds"]
    return t


def probe_ratios(out):
    """{(level, arch): probe seconds at the level / at ref}, median over the probed datasets."""
    parts = []
    for f in sorted((out / "readout_step0").glob("probe_*.csv")):
        d = pd.read_csv(f).groupby(["dataset", "level", "config"]).probe_seconds.sum().reset_index()
        ref = d[d.level == "ref"].set_index(["dataset", "config"]).probe_seconds
        d["ratio"] = [r.probe_seconds / ref[(r.dataset, r.config)] for r in d.itertuples()]
        parts.append(d)
    if not parts:
        raise SystemExit("no readout_step0/probe_*.csv: run step 0 with --probe first")
    return pd.concat(parts).groupby(["level", "config"]).ratio.median().to_dict()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="step (a') results (with step 0 done)")
    ap.add_argument("--step-b", type=Path, required=True, help="step (b) results, e.g. $A/11-9-26")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--safety", type=float, default=1.5)
    a = ap.parse_args()
    ratio = probe_ratios(a.out)
    rows = []
    for ds in DATASETS:
        cfg = dataset_config(ds, None)
        levels = [lv for lv in readout.LEVELS if readout.applies(lv, cfg)]
        ext = extract_seconds(a.out, ds)
        ps = a.step_b / "prep" / "ref" / ds / "per_seed.csv"
        if not ext or not ps.exists():
            print(f"{ds}: step-0 features or step-(b) probes missing, skipped")
            continue
        ps = pd.read_csv(ps)
        ps = ps[ps.kind == "cnn"].groupby(["config", "seed"]).probe_seconds.sum().groupby("config").mean()
        archs = list(ps.index)
        miss = [(lv, ar) for lv in levels for ar in archs if (lv, ar) not in ext or (lv, ar) not in ratio]
        if miss:
            print(f"{ds}: no step-0 measurement for {miss}")
        extract = sum(ext.get((lv, ar), np.nan) for lv in levels for ar in archs) * a.seeds / 3600
        probes = sum(ps[ar] * ratio.get((lv, ar), np.nan) for lv in levels for ar in archs) * a.seeds / 3600
        nf = a.step_b / "prep_summary" / f"nested_{ds}.csv"
        nested = np.nan
        if nf.exists():
            n = pd.read_csv(nf)
            n_prep = len([lv for lv in prep.LEVELS if prep.applies(lv, cfg)])
            mean_ratio = np.nanmean([ratio.get((lv, ar), np.nan) for lv in levels for ar in archs])
            nested = n[n.kind.str.startswith("cnn:")].nested_seconds.sum() / 3600 * len(levels) / n_prep * mean_ratio
        rows.append(dict(dataset=ds, levels=len(levels), extract=extract, probes=probes, nested=nested))
    df = pd.DataFrame(rows).set_index("dataset")
    df.loc["TOTAL"] = df.sum()
    print(df.round(1).to_string())
    total = df.loc["TOTAL", ["extract", "probes", "nested"]].sum()
    print(f"\nmeasured-based total {total:.0f} CPU-h; with safety x{a.safety}: {total * a.safety:.0f} CPU-h "
          "(CPU-hours used; allocation is higher: step (b) used 723 of 1,100 allocated)")


if __name__ == "__main__":
    main()
