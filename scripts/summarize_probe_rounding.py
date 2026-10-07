"""Group A (#30, task 4): collect probe_under_rounding.py outputs and summarize them.

Concatenates <dir>/<kind>/*.csv (kind: probe_only, end_to_end) and computes, per model x
condition, the per-participant SD of P(high BDI) across rounding samples, significant digits
and flip rate, as variability_stats.py does for the other sources.

    python scripts/summarize_probe_rounding.py --dir results/paper/A4_probe_sr --out results/paper/A_stats
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from variability_stats import per_participant  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dir", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

parts, rows = [], []
for kind in ("probe_only", "end_to_end"):
    files = sorted((Path(args.dir) / kind).glob("*.csv"))
    if not files:
        continue
    pred = pd.concat(pd.read_csv(f) for f in files)
    pred["run"] = pred["sr_seed"].astype(str)
    pv = per_participant(pred, f"probe_sr_{kind}")
    parts.append(pv)
    rows.append(pv.groupby(["model", "condition", "source"]).agg(
        n_runs=("n_runs", "first"), median_prob_sd=("prob_sd", "median"),
        median_sig_digits_parker=("sig_digits_parker", "median"), flip_rate=("flip", "mean")).reset_index())
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
pd.concat(parts).to_csv(out / "participant_variability_probe_sr.csv", index=False, float_format="%.6g")
summary = pd.concat(rows)
summary.to_csv(out / "summary_probe_sr.csv", index=False, float_format="%.6g")
print(summary.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
