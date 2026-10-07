"""Group C (#32, task 11): sex and trait anxiety as confounds of the BDI probe.

The groups differ in sex (high BDI 72% female vs 53%) and strongly in trait anxiety (STAI).
For every round-to-nearest weight seed (and the band-power baseline, if given) and each
model and condition, with the same probe and stratified 5-fold split:

  bdi              the main task, for reference
  bdi_sex_resid    sex regressed out of every feature inside each CV fold (fitted on the
                   training participants only), then the probe
  bdi_female       BDI probe on women only
  bdi_male         BDI probe on men only
  stai_median      STAI above vs below its median (same participants) as the target
  sex_only         sex alone as the only feature, predicting BDI (how far sex alone goes)

Writes <out>/runs.csv.gz (one row per run x model x condition x analysis: AUC, balanced
accuracy, n) and <out>/summary.csv (mean, SD over seeds).

    python scripts/confounds.py --config configs/paper.yaml --features results/paper/gpu_features \\
        --weight-seeds 0-29 --bandpower results/paper/C10/bandpower_features.npz --out results/paper/C11
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.cli import seed_range
from untrained_eeg.data import _data_dir, load_config, load_participants
from untrained_eeg.models import MODELS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_runs import participant_features  # noqa: E402


class ResidualizeLastColumn(BaseEstimator, TransformerMixin):
    """Regress every column on the last one (the confound) and keep the residuals of the others."""

    def fit(self, X, y=None):
        self.reg_ = LinearRegression().fit(X[:, -1:], X[:, :-1])
        return self

    def transform(self, X):
        return X[:, :-1] - self.reg_.predict(X[:, -1:])


def oof(X, y, cv_seed, folds, residualize=False):
    steps = ([ResidualizeLastColumn()] if residualize else []) + [StandardScaler(), LogisticRegression(max_iter=5000)]
    cv = StratifiedKFold(folds, shuffle=True, random_state=cv_seed)
    return cross_val_predict(make_pipeline(*steps), X, y, cv=cv, method="predict_proba")[:, 1]


def analyses(X, people, cfg):
    """Every analysis on one feature matrix X (rows aligned with `people`)."""
    seed, k = cfg["cv_seed"], cfg["cv_folds"]
    y, sex = people["label"].to_numpy(), people["female"].to_numpy()
    out = {}

    def score(name, yy, prob):
        out[name] = dict(auc=roc_auc_score(yy, prob), balanced_accuracy=balanced_accuracy_score(yy, prob > 0.5), n=len(yy))

    score("bdi", y, oof(X, y, seed, k))
    score("bdi_sex_resid", y, oof(np.column_stack([X, sex]), y, seed, k, residualize=True))
    for name, m in (("bdi_female", sex == 1), ("bdi_male", sex == 0)):
        score(name, y[m], oof(X[m], y[m], seed, k))
    stai = people["STAI"].to_numpy()
    ok = ~np.isnan(stai)
    ys = (stai[ok] > np.median(stai[ok])).astype(int)
    score("stai_median", ys, oof(X[ok], ys, seed, k))
    score("sex_only", y, oof(sex[:, None].astype(float), y, seed, k))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--features", required=True)
    parser.add_argument("--weight-seeds", type=seed_range, default=[0])
    parser.add_argument("--bandpower", default=None, help="bandpower_features.npz from bandpower_baseline.py")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    people = load_participants(cfg)
    raw = pd.read_csv(_data_dir(cfg) / "participants.tsv", sep="\t")[["participant_id", "sex", "STAI"]]
    people = people.merge(raw, on="participant_id")
    people["female"] = (people["sex"] == 1).astype(int)  # participants.json: 1 = female, 2 = male
    rows = []
    for model in args.models or cfg["models"]:
        for w in args.weight_seeds:
            f = np.load(Path(args.features) / model / f"w{w:03d}_rn.npz")
            for condition in cfg["conditions"]:
                feats = participant_features(f, condition)
                sub = people[people["participant_id"].isin(feats)].reset_index(drop=True)
                X = np.stack([feats[p] for p in sub["participant_id"]])
                for name, r in analyses(X, sub, cfg).items():
                    rows.append(dict(model=model, weight_seed=w, condition=condition, analysis=name, **r))
        print(f"{model}: {len(args.weight_seeds)} seeds done", flush=True)
    if args.bandpower:
        bp = np.load(args.bandpower)
        for condition in cfg["conditions"]:
            sub = people.set_index("participant_id").loc[bp[f"{condition}_participant_id"]].reset_index()
            for name, r in analyses(bp[condition], sub, cfg).items():
                rows.append(dict(model="bandpower", weight_seed=-1, condition=condition, analysis=name, **r))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "runs.csv.gz", index=False, float_format="%.6g")
    summary = runs.groupby(["model", "condition", "analysis"]).agg(
        n_runs=("auc", "size"), n=("n", "first"), auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
        bacc_mean=("balanced_accuracy", "mean")).reset_index()
    summary.to_csv(out / "summary.csv", index=False, float_format="%.4g")
    print(summary.pivot_table(index=["model", "condition"], columns="analysis", values="auc_mean")
          .to_string(float_format=lambda v: f"{v:.3f}"))
