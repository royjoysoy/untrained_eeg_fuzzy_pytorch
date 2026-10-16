"""Group C (#32, task 8): positive control, eyes closed vs eyes open per 1-minute block.

Same saved features as group A (extract_gpu_features.py), but the probe classifies each
participant-block as eyes closed (1) or open (0). CV is stratified by condition and grouped
by participant (StratifiedGroupKFold), so a participant is never in both train and test.
If the random-CNN features carry EEG information, this task should be easy (alpha rises
with eyes closed); the seed vs rounding comparison is then repeated on a task that works.

Writes predictions.csv.gz / runs.csv.gz in the tidy format of probe_runs.py, with
participant_id = "<participant>_b<block>" and label = eyes closed, so variability_stats.py
applies unchanged.

    python scripts/positive_control.py --config configs/paper.yaml \\
        --features results/paper/gpu_features --out results/paper/C8 \\
        --only $(printf 'w%03d_rn ' $(seq 0 29)) $(printf 'w000_sr%03d ' $(seq 0 29))
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.cli import seed_range
from untrained_eeg.data import load_config, load_participants
from untrained_eeg.models import MODELS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_runs import metrics  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default=None)
parser.add_argument("--features", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--models", nargs="+", choices=MODELS, default=None)
parser.add_argument("--cv-seeds", type=seed_range, default=None)
parser.add_argument("--only", nargs="+", default=None, help="run names to keep, e.g. w000_rn")
args = parser.parse_args()

cfg = load_config(args.config) if args.config else load_config()
keep = set(load_participants(cfg)["participant_id"])  # labelled participants only, as in the main task
cv_seeds = args.cv_seeds if args.cv_seeds is not None else [cfg["cv_seed"]]
pred_rows, run_rows = [], []
for model in args.models or cfg["models"]:
    files = sorted((Path(args.features) / model).glob("w*.npz"))
    if args.only:
        files = [f for f in files if f.stem in args.only]
    for path in files:
        f = np.load(path)
        m = np.isin(f["participant_id"], list(keep))
        X, groups = f["block_feats"][m], f["participant_id"][m]
        y = (f["cond"][m] == "closed").astype(int)
        ids = np.char.add(np.char.add(groups.astype(str), "_b"), f["block"][m].astype(str))
        run = dict(model=model, source="seed" if str(f["mode"]) == "rn" else "sr",
                   weight_seed=int(f["weight_seed"]), sr_seed=int(f["sr_seed"]), condition="eyes_closed_vs_open")
        for cv_seed in cv_seeds:
            cv = StratifiedGroupKFold(cfg["cv_folds"], shuffle=True, random_state=cv_seed)
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
            prob = cross_val_predict(clf, X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
            key = dict(run, cv_seed=cv_seed)
            run_rows.append(dict(key, n=len(y), **metrics(y, prob)))
            pred_rows.append(pd.DataFrame(dict(key, participant_id=ids, label=y, prob=prob)))
        print(f"{model} {path.stem}: AUC {run_rows[-1]['auc']:.4f}", flush=True)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
pd.concat(pred_rows).to_csv(out / "predictions.csv.gz", index=False, float_format="%.17g")
pd.DataFrame(run_rows).to_csv(out / "runs.csv.gz", index=False, float_format="%.10g")
print(f"wrote {out}")
