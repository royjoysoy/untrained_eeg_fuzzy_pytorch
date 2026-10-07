"""Group A (#30): probe every saved feature run (extract_gpu_features.py) and write tidy results.

For each run file <features>/<model>/w<WWW>_<rn|srKKK>.npz and each condition, participant
features are the window-weighted mean of their block features (= mean over windows), then
the usual probe predicts high vs low BDI: standardize + logistic regression, stratified
5-fold CV. Writes, gzip-compressed:

  <out>/predictions.csv.gz   one row per participant x run x model x condition x CV split:
                             participant_id, label, model, condition, source, weight_seed,
                             sr_seed, cv_seed, prob (out-of-fold P(high BDI))
  <out>/runs.csv.gz          one row per run x model x condition x CV split: AUC, accuracy,
                             balanced accuracy, F1, precision, recall, specificity

source: "seed" (round-to-nearest run of a weight seed) or "sr" (stochastic rounding).

    python scripts/probe_runs.py --features results/paper/gpu_features --out results/paper/A1
    python scripts/probe_runs.py ... --cv-seeds 0-29 --only w000_rn   # A4: data-split variability
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.cli import seed_range
from untrained_eeg.data import load_config, load_participants
from untrained_eeg.models import MODELS

THRESHOLD = 0.5


def participant_features(f, condition):
    """{participant_id: window-weighted mean of its block features in `condition`}."""
    keep = f["cond"] == condition
    feats, w, pid = f["block_feats"][keep], f["n_windows"][keep], f["participant_id"][keep]
    out = {}
    for p in np.unique(pid):
        m = pid == p
        out[p] = (feats[m] * w[m, None]).sum(axis=0) / w[m].sum()
    return out


def probe(X, y, cv_seed, folds):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    cv = StratifiedKFold(folds, shuffle=True, random_state=cv_seed)
    return cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]


def metrics(y, prob):
    pred = (prob > THRESHOLD).astype(int)
    return dict(auc=roc_auc_score(y, prob), accuracy=accuracy_score(y, pred),
                balanced_accuracy=balanced_accuracy_score(y, pred), f1=f1_score(y, pred, zero_division=0),
                precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred, zero_division=0),
                specificity=recall_score(y, pred, pos_label=0, zero_division=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--features", required=True, help="folder written by extract_gpu_features.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=None)
    parser.add_argument("--cv-seeds", type=seed_range, default=None, help="default: cv_seed of the config")
    parser.add_argument("--only", nargs="+", default=None, help="run names to keep, e.g. w000_rn")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    labels = load_participants(cfg).set_index("participant_id")["label"]
    cv_seeds = args.cv_seeds if args.cv_seeds is not None else [cfg["cv_seed"]]
    pred_rows, run_rows = [], []
    for model in args.models or cfg["models"]:
        files = sorted((Path(args.features) / model).glob("w*.npz"))
        if args.only:
            files = [f for f in files if f.stem in args.only]
        for path in files:
            f = np.load(path)
            source = "seed" if str(f["mode"]) == "rn" else "sr"
            run = dict(model=model, source=source, weight_seed=int(f["weight_seed"]), sr_seed=int(f["sr_seed"]))
            for condition in cfg["conditions"]:
                feats = participant_features(f, condition)
                pids = [p for p in labels.index if p in feats]
                X, y = np.stack([feats[p] for p in pids]), labels[pids].to_numpy()
                for cv_seed in cv_seeds:
                    prob = probe(X, y, cv_seed, cfg["cv_folds"])
                    key = dict(run, condition=condition, cv_seed=cv_seed)
                    run_rows.append(dict(key, n=len(y), **metrics(y, prob)))
                    pred_rows.append(pd.DataFrame(dict(key, participant_id=pids, label=y, prob=prob)))
            print(f"{model} {path.stem}: AUC " + " ".join(
                f"{r['condition']} {r['auc']:.4f}" for r in run_rows[-len(cfg['conditions']) * len(cv_seeds):]
                if r["cv_seed"] == cv_seeds[0]), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(pred_rows).to_csv(out / "predictions.csv.gz", index=False, float_format="%.17g")
    pd.DataFrame(run_rows).to_csv(out / "runs.csv.gz", index=False, float_format="%.10g")
    print(f"wrote {out}/predictions.csv.gz and runs.csv.gz ({len(run_rows)} probe fits)")
