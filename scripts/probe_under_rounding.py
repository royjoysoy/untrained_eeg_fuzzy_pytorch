"""Group A (#30, task 4): run the probe itself under stochastic rounding (Fuzzy PyTorch, CPU).

scikit-learn is never instrumented, so this re-implements the probe in PyTorch with the same
objective as sklearn's default LogisticRegression (L2, C=1, unpenalized intercept), after the
same standardization, fitted with L-BFGS in float64, on the same stratified 5-fold split.
Run it inside the Fuzzy build: every float operation of the probe is then randomly rounded.

  --features-run w000_rn      probe-only perturbation (features from round-to-nearest)
  --features-run w000_srKKK   end to end (features from GPU stochastic rounding run KKK)

One process = one rounding sample (the Fuzzy build seeds rounding per process):

    for k in $(seq 0 29); do
      MODE=sr SEED=$((2001 + k)) bash environment/fuzzy_arm.sh scripts/probe_under_rounding.py \\
        --config cfg.yaml --features results/paper/gpu_features --features-run w000_rn \\
        --sample $k --out results/paper/A4_probe_sr
    done

Writes <out>/<model>_<condition>_<features-run>_sample<KKK>.csv in the tidy format of
probe_runs.py (source "probe_sr"); concatenate them for variability_stats.py.
Outside the Fuzzy build (MODE=rn or plain PyTorch) every sample must be identical.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.data import load_config, load_participants
from untrained_eeg.models import MODELS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_runs import participant_features  # noqa: E402


def fit_predict(X_train, y_train, X_test, C=1.0):
    """Standardize on the training fold, fit L2 logistic regression, return P(class 1) on test."""
    mu, sd = X_train.mean(0), X_train.std(0, unbiased=False)
    sd = torch.where(sd == 0, torch.ones_like(sd), sd)
    Xt, Xs = (X_train - mu) / sd, (X_test - mu) / sd
    w = torch.zeros(Xt.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=5000, tolerance_grad=1e-10, tolerance_change=1e-14,
                            history_size=20, line_search_fn="strong_wolfe")

    def loss():  # sklearn objective: C * sum(log loss) + 0.5 * ||w||^2
        opt.zero_grad()
        z = Xt @ w + b
        value = C * torch.nn.functional.softplus(-(2 * y_train - 1) * z).sum() + 0.5 * (w * w).sum()
        value.backward()
        return value

    opt.step(loss)
    with torch.no_grad():
        return torch.sigmoid(Xs @ w + b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--features", required=True)
    parser.add_argument("--features-run", required=True, help="e.g. w000_rn or w000_sr003")
    parser.add_argument("--sample", type=int, required=True, help="rounding sample index (file name)")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)
    cfg = load_config(args.config) if args.config else load_config()
    labels = load_participants(cfg).set_index("participant_id")["label"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for model in args.models or cfg["models"]:
        f = np.load(Path(args.features) / model / f"{args.features_run}.npz")
        for condition in cfg["conditions"]:
            feats = participant_features(f, condition)
            pids = [p for p in labels.index if p in feats]
            X = torch.from_numpy(np.stack([feats[p] for p in pids]).astype(np.float64))
            y = labels[pids].to_numpy()
            yt = torch.from_numpy(y.astype(np.float64))
            prob = np.zeros(len(y))
            cv = StratifiedKFold(cfg["cv_folds"], shuffle=True, random_state=cfg["cv_seed"])
            for train, test in cv.split(np.zeros(len(y)), y):
                prob[test] = fit_predict(X[train], yt[train], X[test]).numpy()
            pd.DataFrame(dict(model=model, source="probe_sr", weight_seed=int(f["weight_seed"]),
                              sr_seed=args.sample, cv_seed=cfg["cv_seed"], condition=condition,
                              features_run=args.features_run, participant_id=pids, label=y, prob=prob)).to_csv(
                out / f"{model}_{condition}_{args.features_run}_sample{args.sample:03d}.csv",
                index=False, float_format="%.17g")
            print(f"{model} {condition} {args.features_run} sample {args.sample}: mean P {prob.mean():.10f}",
                  flush=True)
