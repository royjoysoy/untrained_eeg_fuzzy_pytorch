"""Sensitivity of the phase 1 results to the probe: does one C for every extractor tilt the
comparison? Extractors give 64 to ~14,000 features for 31 to 202 participants, so C = 1 is a
very different regularization for each. Two alternative probes, fitted in training folds only:

  nested   C chosen in each training fold by an inner CV (C = 1e-4 .. 1e2)
  pca      standardized features reduced to the same number of whitened components for every
           extractor (32, or the training fold size - 1 if smaller), then C = 1

Probes every instance's saved features (phase1_features.py) with the config's CV split, the
same outer folds as the main probe (src/untrained_eeg/tasks.py, PROBES). Writes
results/sensitivity/<probe>/<dataset>/metrics.csv (with the median chosen C for nested) and
predictions.csv.gz, in the format of phase1_probe.py, so phase 2 and ranking.py can read them.
Run it as a job (nested is about 35 times the main probe):
    python scripts/probe_sensitivity.py --config configs/ds002778.yaml --probe pca
--extractors probes only those (e.g. a model added later) and keeps the other extractors' saved rows.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.data import load_config
from untrained_eeg.tasks import probe, task_data, task_labels, task_names


def run(ext, path, tasks, labels, cfg, variant):
    """Metrics and predictions of one instance on every task."""
    f = np.load(path)
    metrics, preds = [], []
    for task in tasks:
        X, y, groups, ids = task_data(f["feats"], f["cond"], f["participant_id"], task, labels)
        chosen = []
        prob, auc, bacc = probe(X, y, groups, task, cfg["cv_seed"], cfg["cv_folds"], variant, chosen)
        metrics.append(dict(extractor=ext, group=str(f["group"]), instance=path.stem, task=task, probe=variant,
                            cv_seed=cfg["cv_seed"], n=len(y), n_features=X.shape[1], auc=auc,
                            balanced_accuracy=bacc, c_median=np.median(chosen) if chosen else np.nan))
        cols = {"prob": prob} if prob.ndim == 1 else {f"prob_{k}": prob[:, k] for k in range(prob.shape[1])}
        preds.append(pd.DataFrame(dict(extractor=ext, instance=path.stem, task=task, cv_seed=cfg["cv_seed"],
                                       sample=ids, label=y, **cols)))
    return metrics, preds


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default=None)
parser.add_argument("--probe", choices=["nested", "pca"], required=True)
parser.add_argument("--extractors", nargs="+", default=None, help="only these; keep the others' saved results")
parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
args = parser.parse_args()

cfg = load_config(args.config) if args.config else load_config()
ds = cfg["openneuro"]["id"]
src = Path(cfg["results_dir"]) / "phase1" / ds / "features"
dest = Path(cfg["results_dir"]) / "sensitivity" / args.probe / ds
dest.mkdir(parents=True, exist_ok=True)
labels, tasks = task_labels(cfg), task_names(cfg)
exts = sorted(d for d in src.iterdir() if d.is_dir() and (args.extractors is None or d.name in args.extractors))
jobs = [(e.name, p) for e in exts for p in sorted(e.glob("*.npz"))]
out = Parallel(n_jobs=args.workers, verbose=5)(delayed(run)(e, p, tasks, labels, cfg, args.probe) for e, p in jobs)
metrics = pd.DataFrame([m for ms, _ in out for m in ms])
preds = pd.concat([p for _, ps in out for p in ps])
if args.extractors:  # add to the saved results of the other extractors
    keep = lambda df: df[~df.extractor.isin(args.extractors)]
    metrics = pd.concat([keep(pd.read_csv(dest / "metrics.csv")), metrics])
    preds = pd.concat([keep(pd.read_csv(dest / "predictions.csv.gz")), preds])
metrics.to_csv(dest / "metrics.csv", index=False, float_format="%.6g")
preds.to_csv(dest / "predictions.csv.gz", index=False, float_format="%.6g")
summary = metrics.groupby(["task", "extractor"]).agg(auc=("auc", "median"), c=("c_median", "median"))
print(summary.to_string(float_format="%.3f"))
print(f"wrote {dest}")
