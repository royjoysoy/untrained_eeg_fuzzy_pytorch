"""Classification metrics and stability: seeds (02) vs numerical samples (03).

Reads results/seed/, results/mca/ (Fuzzy PyTorch, CPU) and results/turbulence/
(GPU), whichever exist: the per-run out-of-fold P(high BDI) and
labels), recomputes the usual classification metrics for every run, and writes
results/metrics_summary.csv:

  one row per source (seed / fuzzy_cpu / turbulence) x model x condition, with mean, std, min, max
  across runs of accuracy, balanced accuracy, F1, precision, recall, specificity,
  ROC-AUC, plus per-participant stability of P(high BDI):
    prob_std     mean over participants of the std across runs
    sig_digits   mean significant digits shared by the runs (15.9 = identical)
    flip_rate    fraction of participants whose predicted class changes across runs

    python scripts/summarize_metrics.py --config configs/ds003478.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.data import load_config
from untrained_eeg.variability import significant_digits

THRESHOLD = 0.5  # P(high BDI) > 0.5 -> predicted high
# results sub-folder -> label. mca: Fuzzy PyTorch on CPU (03_mca_variability.py),
# turbulence: stochastic rounding on GPU (03_turbulence_variability.py).
SOURCES = {"seed": "seed", "mca": "fuzzy_cpu", "turbulence": "turbulence"}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default=None)
args = parser.parse_args()
cfg = load_config(args.config) if args.config else load_config()
res = cfg["results_dir"]


def load_runs(source, model, condition):
    """prob (n_runs, n_participants) and labels, or None if missing."""
    if source == "seed":
        f = res / "seed" / f"{model}_{condition}.npz"
        if not f.exists():
            return None
        d = np.load(f)
        return d["prob"], d["label"]
    files = sorted((res / source / f"{model}_{condition}").glob("sample-*.npz"))
    if not files:
        return None
    # allow_pickle: 03_turbulence_variability.py saves participant ids as an object array
    runs = [np.load(f, allow_pickle=True) for f in files]
    ids = runs[0]["participant_id"].astype(str)
    for r in runs[1:]:  # all samples must describe the same participants
        assert np.array_equal(r["participant_id"].astype(str), ids)
    return np.stack([r["prob"] for r in runs]), runs[0]["label"]


def run_metrics(prob, y):
    pred = (prob > THRESHOLD).astype(int)
    return dict(
        accuracy=accuracy_score(y, pred),
        balanced_accuracy=balanced_accuracy_score(y, pred),
        f1=f1_score(y, pred, zero_division=0),
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        specificity=recall_score(y, pred, pos_label=0, zero_division=0),
        auc=roc_auc_score(y, prob),
    )


rows = []
for model in cfg["models"]:
    for condition in cfg["conditions"]:
        for source in SOURCES:
            runs = load_runs(source, model, condition)
            if runs is None or len(runs[0]) < 2:
                continue
            prob, y = runs
            per_run = pd.DataFrame([run_metrics(p, y) for p in prob])
            pred = prob > THRESHOLD
            row = dict(source=SOURCES[source], model=model, condition=condition, n_runs=len(prob))
            for k in per_run:
                row[f"{k}_mean"] = per_run[k].mean()
                row[f"{k}_std"] = per_run[k].std(ddof=1)
                row[f"{k}_min"] = per_run[k].min()
                row[f"{k}_max"] = per_run[k].max()
            row["prob_std"] = prob.std(axis=0, ddof=1).mean()
            row["sig_digits"] = significant_digits(prob).mean()
            row["flip_rate"] = (pred.any(axis=0) & ~pred.all(axis=0)).mean()
            rows.append(row)

if not rows:
    raise SystemExit("No results yet. Run scripts 02 and 03 first.")
table = pd.DataFrame(rows)
table.to_csv(res / "metrics_summary.csv", index=False, float_format="%.6g")
print(f"wrote {res / 'metrics_summary.csv'}\n")

# Short view: mean +/- std for the main metrics, then stability
show = table[["source", "model", "condition", "n_runs"]].copy()
for k in ("accuracy", "balanced_accuracy", "f1", "auc"):
    show[k] = [f"{m:.3f} ± {s:.1e}" if s < 1e-3 else f"{m:.3f} ± {s:.3f}"
               for m, s in zip(table[f"{k}_mean"], table[f"{k}_std"])]
show["prob_std"] = table["prob_std"].map("{:.1e}".format)
show["sig_digits"] = table["sig_digits"].map("{:.1f}".format)
show["flip_rate"] = table["flip_rate"].map("{:.1%}".format)
print(show.sort_values(["model", "condition", "source"]).to_string(index=False))
