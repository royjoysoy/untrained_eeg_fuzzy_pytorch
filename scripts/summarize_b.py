"""Group B (#31): summarize the precision sweep (B5) and the per-layer perturbation (B6).

Inputs are probe_runs.py outputs (predictions.csv.gz):
  --seed-probe   group A run with the round-to-nearest seeds (reference: seed variability)
  --precision    t=DIR pairs, one per precision, e.g. 24=results/paper/A1 17=results/paper/B5/p17_probe
                 (for t=24 the A1 stochastic-rounding runs of weight seed 0 are used)
  --layers       DIR, one probe output per <model>_<layer> run, e.g. results/paper/B6/*_probe

Writes <out>/precision_sweep.csv, <out>/per_layer.csv and <out>/precision_sweep.png:
median over participants of the SD of P(high BDI) across rounding samples, per model,
condition and precision, with the seed SD as a horizontal reference.

    python scripts/summarize_b.py --seed-probe results/paper/A1 \\
        --precision 24=results/paper/A1 17=results/paper/B5/p17_probe 11=... \\
        --layers results/paper/B6/*_probe --out results/paper/B_summary
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from variability_stats import load, per_participant  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--seed-probe", required=True)
parser.add_argument("--precision", nargs="+", default=[])
parser.add_argument("--layers", nargs="*", default=[])
parser.add_argument("--out", required=True)
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

seed = per_participant(load(args.seed_probe).query("source == 'seed'"), "seed")
seed_ref = seed.groupby(["model", "condition"])["prob_sd"].median().rename("seed_median_sd")

rows = []
for item in args.precision:
    t, folder = item.split("=", 1)
    pred = load(folder).query("source == 'sr' and weight_seed == 0")
    pv = per_participant(pred, f"t{t}")
    for (model, condition), g in pv.groupby(["model", "condition"]):
        rows.append(dict(model=model, condition=condition, precision=int(t), n_runs=int(g["n_runs"].iloc[0]),
                         median_sd=g["prob_sd"].median(), q90_sd=g["prob_sd"].quantile(0.9),
                         median_sig_digits=g["sig_digits_parker"].median(), flip_rate=g["flip"].mean()))
if rows:
    sweep = pd.DataFrame(rows).merge(seed_ref.reset_index(), on=["model", "condition"])
    sweep["ratio_to_seed"] = sweep["median_sd"] / sweep["seed_median_sd"]
    sweep.sort_values(["model", "condition", "precision"]).to_csv(out / "precision_sweep.csv", index=False,
                                                                   float_format="%.6g")
    print(sweep.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    conds = sorted(sweep["condition"].unique())
    fig, axes = plt.subplots(1, len(conds), figsize=(4.2 * len(conds), 3.6), sharey=True)
    colors = {"cnn1d": "#1b6e8a", "shallow": "#c9731b", "eegnet": "#6a51a3"}
    for ax, condition in zip(np.atleast_1d(axes), conds):
        for model, g in sweep[sweep.condition == condition].groupby("model"):
            g = g.sort_values("precision")
            ax.plot(g["precision"], g["median_sd"], "o-", color=colors.get(model), label=model)
            ax.axhline(g["seed_median_sd"].iloc[0], color=colors.get(model), ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_xticks(sorted(sweep["precision"].unique()))
        ax.set_xlabel("significant bits (24 = float32, 11 = fp16, 8 = bf16)")
        ax.set_title(f"eyes {condition}")
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("median SD of P(high BDI)\n(dashed: across seeds)")
    np.atleast_1d(axes)[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "precision_sweep.png", dpi=200)
    print(f"wrote {out / 'precision_sweep.png'}")

layer_rows = []
for folder in args.layers:
    pred = load(folder).query("source == 'sr' and weight_seed == 0")
    if pred.empty:
        continue
    model, layer = Path(folder).name.removesuffix("_probe").split("_", 1)
    pv = per_participant(pred, layer)
    for (m, condition), g in pv.groupby(["model", "condition"]):
        layer_rows.append(dict(model=m, layer=layer, condition=condition, n_runs=int(g["n_runs"].iloc[0]),
                               median_sd=g["prob_sd"].median(), median_sig_digits=g["sig_digits_parker"].median(),
                               flip_rate=g["flip"].mean()))
if layer_rows:
    layers = pd.DataFrame(layer_rows)
    layers.to_csv(out / "per_layer.csv", index=False, float_format="%.6g")
    print(layers.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
