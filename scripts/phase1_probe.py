"""Phase 1, step 2: probe every instance's features on a dataset's tasks.

Tasks and probe: src/untrained_eeg/tasks.py (per dataset; ds003478 by default).

Every instance (seed or checkpoint) is probed with the config's CV split. The first
instance of each extractor is also probed with --cv-seeds, the CV-split variability.

Reads results/phase1/<dataset>/features (phase1_features.py). Writes, next to it:
  metrics.csv            extractor, group, instance, task, cv_seed, n, auc, balanced_accuracy
  predictions.csv.gz     out-of-fold P(class 1) per sample (prob_<k> per class for a
                         multi-class task), for the ensembles of phase 2

    python scripts/phase1_probe.py                                   # ds003478
    python scripts/phase1_probe.py --config configs/ds004504.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.data import load_config
from untrained_eeg.tasks import probe, task_data, task_labels, task_names


def seed_range(text):
    a, _, b = text.partition("-")
    return list(range(int(a), int(b or a) + 1))


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default=None)
parser.add_argument("--cv-seeds", type=seed_range, default=list(range(30)),
                    help="CV splits for the first instance of each extractor, e.g. 0-29")
args = parser.parse_args()

cfg = load_config(args.config) if args.config else load_config()
if "cv_split_seeds" in cfg:  # a dataset whose probes are slow (sleep staging) uses fewer
    args.cv_seeds = args.cv_seeds[: cfg["cv_split_seeds"]]
if cfg["cv_seed"] not in args.cv_seeds:  # the first instance also needs the fixed split
    args.cv_seeds = [cfg["cv_seed"], *args.cv_seeds]
root = Path(cfg["results_dir"]) / "phase1" / cfg["openneuro"]["id"]
labels = task_labels(cfg)
metrics, preds = [], []
for ext_dir in sorted(p for p in (root / "features").iterdir() if p.is_dir()):
    start = len(metrics)  # this extractor's rows start here
    for i, path in enumerate(sorted(ext_dir.glob("*.npz"))):
        f = np.load(path)
        cv_seeds = args.cv_seeds if i == 0 else [cfg["cv_seed"]]
        for task in task_names(cfg):
            X, y, groups, ids = task_data(f["feats"], f["cond"], f["participant_id"], task, labels)
            for cv_seed in cv_seeds:
                prob, auc, bacc = probe(X, y, groups, task, cv_seed, cfg["cv_folds"])
                metrics.append(dict(extractor=ext_dir.name, group=str(f["group"]), instance=path.stem,
                                    task=task, cv_seed=cv_seed, n=len(y), auc=auc, balanced_accuracy=bacc))
                # P(class 1), or one column prob_<k> per class for a multi-class task
                cols = {"prob": prob} if prob.ndim == 1 else {f"prob_{k}": prob[:, k] for k in range(prob.shape[1])}
                preds.append(pd.DataFrame(dict(extractor=ext_dir.name, instance=path.stem, task=task,
                                               cv_seed=cv_seed, sample=ids, label=y, **cols)))
    m = pd.DataFrame(metrics[start:])
    m = m[m.cv_seed == cfg["cv_seed"]]
    for task, g in m.groupby("task"):
        print(f"{ext_dir.name:18s} {task:10s} {len(g):3d} instances: AUC {g.auc.mean():.3f} ± {g.auc.std():.3f}",
              flush=True)

pd.DataFrame(metrics).to_csv(root / "metrics.csv", index=False, float_format="%.6g")
pd.concat(preds).to_csv(root / "predictions.csv.gz", index=False, float_format="%.6g")
print(f"wrote {root / 'metrics.csv'}")
