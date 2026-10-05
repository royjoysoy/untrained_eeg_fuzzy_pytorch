"""Goal 4 (advanced): compare seed variability (02) with MCA variability (03).

Reads results/seed/ and results/mca/, writes:
  results/summary.csv               one row per model x condition x source
  results/figures/auc.png           probe AUC of every run
  results/figures/sig_digits.png    significant digits of P(high BDI) per participant

    python scripts/04_compare.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # write files, no window
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.data import load_config
from untrained_eeg.variability import significant_digits, summarize

COLORS = {"seed": "#2a78d6", "mca": "#eb6834"}  # blue / orange, colorblind-safe pair
LABELS = {"seed": "Seeds (random init)", "mca": "MCA (Fuzzy PyTorch)"}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default=None)
args = parser.parse_args()
cfg = load_config(args.config) if args.config else load_config()
res = cfg["results_dir"]


def load_runs(source, model, condition):
    """Stack all runs of one source into prob (n_runs, n_participants) and auc (n_runs,)."""
    if source == "seed":
        files = [res / "seed" / f"{model}_{condition}.npz"]
    else:
        files = sorted((res / "mca" / f"{model}_{condition}").glob("sample-*.npz"))
    files = [f for f in files if f.exists()]
    if not files:
        return None
    runs = [np.load(f) for f in files]
    prob = np.vstack([np.atleast_2d(r["prob"]) for r in runs])
    auc = np.concatenate([np.atleast_1d(r["auc"]) for r in runs])
    return prob, auc


rows, data = [], {}
for model in cfg["models"]:
    for condition in cfg["conditions"]:
        for source in ("seed", "mca"):
            runs = load_runs(source, model, condition)
            if runs is None or len(runs[1]) < 2:
                print(f"skipping {source} {model} {condition}: fewer than 2 runs")
                continue
            data[source, model, condition] = runs
            rows.append(dict(source=source, model=model, condition=condition, **summarize(*runs)))

if not rows:
    raise SystemExit("No results yet. Run scripts 02 and 03 first.")
summary = pd.DataFrame(rows)
summary.to_csv(res / "summary.csv", index=False, float_format="%.6g")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

# ---- Figures: one panel per condition, models on the x axis -------------
fig_dir = res / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
rng = np.random.default_rng(0)  # only for horizontal jitter of the dots


def strip_figure(value_fn, ylabel, filename, ylim=None):
    fig, axes = plt.subplots(1, len(cfg["conditions"]), figsize=(7, 3.5), sharey=True)
    for ax, condition in zip(np.atleast_1d(axes), cfg["conditions"]):
        for i, model in enumerate(cfg["models"]):
            for j, source in enumerate(("seed", "mca")):
                if (source, model, condition) not in data:
                    continue
                y = value_fn(*data[source, model, condition])
                x = i + (j - 0.5) * 0.36 + rng.uniform(-0.1, 0.1, len(y))
                ax.scatter(x, y, s=10, color=COLORS[source], alpha=0.6, linewidths=0,
                           label=LABELS[source] if i == 0 else None)
                ax.hlines(np.median(y), i + (j - 0.5) * 0.36 - 0.15, i + (j - 0.5) * 0.36 + 0.15,
                          color="#0b0b0b", linewidth=2)
        ax.set_xticks(range(len(cfg["models"])), cfg["models"])
        ax.set_title(f"eyes {condition}")
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        if ylim:
            ax.set_ylim(*ylim)
    np.atleast_1d(axes)[0].set_ylabel(ylabel)
    fig.legend(*np.atleast_1d(axes)[0].get_legend_handles_labels(), loc="upper center",
               ncol=2, frameon=False, markerscale=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(fig_dir / filename, dpi=200)
    print(f"wrote {fig_dir / filename}")


# one dot per run
strip_figure(lambda prob, auc: auc, "probe AUC (high vs low BDI)", "auc.png")
# one dot per participant; 15.9 = all runs agree to the last float64 digit
strip_figure(lambda prob, auc: significant_digits(prob), "significant digits of P(high BDI)",
             "sig_digits.png", ylim=(0, 16.5))
