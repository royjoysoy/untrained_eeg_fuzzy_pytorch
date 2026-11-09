#!/usr/bin/env python
"""CPU-hour estimate for step (b) from what step (a) actually used (no new runs needed).

    python optimization/prep_estimate.py --step-a $A/11-5-26 $A/11-5-26b [--safety 1.5]

Measured in step (a), per dataset: CNN extraction of `default` and of the combined config
(run_info JSONs: wall seconds x CPUs), handcrafted extraction, MiniRocket, probe seconds per
feature file (per_seed*.csv) and nested-selection seconds (nested.csv). Step (b) repeats each
once per level that applies (prep.applies), with cost multipliers for the inputs that change
the work: 256 Hz = 2x samples (sf256_k2: also a 2x first kernel, CNN ~4x); nested selection
scales with the number of candidates (levels instead of 31 grid configs). Everything is
multiplied by --safety (step (a) used about 2x my estimate). Prints CPU-hours per stage.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimization import prep  # noqa: E402
from optimization.common import DATASETS, dataset_config  # noqa: E402

COMBINED = "cov_channels=32,n_filters=256,readout=meanstd_ppv_max"
GRID = 31  # candidates in step (a)'s nested selection


def cpu_hours(rec):
    cpus = int(rec.get("slurm", {}).get("SLURM_CPUS_PER_TASK", 1))
    return rec["wall_seconds"] * cpus / 3600


def measured(dirs):
    """{(dataset, stage): CPU-h} from step (a)'s run records (successful runs only)."""
    out = {}
    for d in dirs:
        for f in (d / "run_info").glob("*.json"):
            r = json.loads(f.read_text())
            if r.get("status") != "ok" or not r.get("dataset"):
                continue
            cfgs = r.get("configs") or []
            stage = {"extract_handcrafted": "handcrafted", "minirocket": "minirocket"}.get(r["name"])
            if r["name"] == "extract_cnn" and cfgs == ["default"]:
                stage = "cnn_default"
            elif r["name"] == "extract_cnn" and cfgs == [COMBINED]:
                stage = "cnn_combined"
            if stage:
                key = (r["dataset"], stage)
                out[key] = out.get(key, 0) + cpu_hours(r)  # search chunks add up to 30 seeds
    return out


def read_all(dirs, ds, pattern, cols):
    files = [f for d in dirs for f in sorted((d / ds).glob(pattern))]
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame(columns=cols)


def factor(level, stage):
    p = prep.LEVELS[level]
    if p.variant == "fm256":
        return 4.0 if (p.kernel_x2 and stage.startswith("cnn")) else 2.0
    return 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step-a", type=Path, nargs="+", required=True)
    ap.add_argument("--safety", type=float, default=1.5)
    a = ap.parse_args()
    m = measured(a.step_a)
    rows = []
    for ds in DATASETS:
        cfg = dataset_config(ds, None)
        lv = [x for x in prep.LEVELS if prep.applies(x, cfg)]
        lv_h = [x for x in lv if prep.for_handcrafted(x)]
        lv_m = [x for x in lv if prep.for_minirocket(x)]
        ps = read_all(a.step_a, ds, "per_seed*.csv", ["config", "kind", "probe_seconds"])
        ns = read_all(a.step_a, ds, "nested*.csv", ["kind", "nested_seconds"])
        if ps.empty:
            print(f"{ds}: no step-(a) results found, skipped")
            continue
        probe = {k: ps[ps.config == k].probe_seconds.sum() / 3600 for k in ("default", COMBINED)}
        probe_h = ps[ps.kind == "handcrafted"].probe_seconds.sum() / 3600
        nest_grid = ns[ns.kind == "cnn_grid"].nested_seconds.sum() / 3600
        nest_h = ns[ns.kind == "handcrafted"].nested_seconds.sum() / 3600
        ratio = probe[COMBINED] / probe["default"] if probe["default"] else 1.0  # bigger features, slower probe
        r = dict(dataset=ds, levels=len(lv),
                 cnn=sum((m.get((ds, "cnn_default"), 0) + m.get((ds, "cnn_combined"), 0)) * factor(x, "cnn") for x in lv),
                 handcrafted=sum(m.get((ds, "handcrafted"), 0) * factor(x, "h") for x in lv_h),
                 minirocket=sum(m.get((ds, "minirocket"), 0) * factor(x, "m") for x in lv_m),
                 probes=len(lv) * (probe["default"] + probe[COMBINED]) + len(lv_h) * probe_h,
                 nested=nest_grid * len(lv) / GRID * (1 + ratio) + nest_h * len(lv_h))
        miss = [s for s in ("cnn_default", "cnn_combined", "handcrafted", "minirocket") if (ds, s) not in m]
        r["missing_records"] = " ".join(miss)
        rows.append(r)
        print(ds, {k: round(v, 1) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    df = pd.DataFrame(rows).set_index("dataset")
    num = df.select_dtypes("number").drop(columns="levels")
    df.loc["TOTAL"] = num.sum()
    print(df.round(1).to_string())
    total = num.sum().sum()
    print(f"\nmeasured-based total {total:.0f} CPU-h; with safety x{a.safety}: {total * a.safety:.0f} CPU-h")


if __name__ == "__main__":
    main()
