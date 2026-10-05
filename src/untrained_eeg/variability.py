"""Untrained-CNN features -> logistic-regression probe -> variability across runs.

One "run" = build a random CNN, embed every window, average the windows of each
participant, and classify high vs low BDI with a cross-validated probe.
Variability is measured on the probe outputs (per-participant probabilities, AUC)
across runs that differ by seed (02) or by MCA sample (03).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .models import build_model, embed


def participant_features(model, windows, participant_ids, condition):
    """Mean feature vector per participant over their `condition` windows."""
    feats = []
    for pid in participant_ids:
        X, cond = windows[pid]
        feats.append(embed(model, X[cond == condition]).mean(axis=0))
    return np.stack(feats)


def probe(features, labels, cfg):
    """Out-of-fold P(high BDI) per participant, plus AUC and balanced accuracy."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    cv = StratifiedKFold(cfg["cv_folds"], shuffle=True, random_state=cfg["cv_seed"])
    prob = cross_val_predict(clf, features, labels, cv=cv, method="predict_proba")[:, 1]
    return prob, roc_auc_score(labels, prob), balanced_accuracy_score(labels, prob > 0.5)


def usable(participants, windows, condition):
    """Participants that have at least one window of `condition`."""
    keep = [(windows[p][1] == condition).any() for p in participants["participant_id"]]
    return participants[keep].reset_index(drop=True)


def run_once(cfg, participants, windows, model_name, condition, seed):
    """One full run. Returns a dict of NumPy arrays, ready for np.savez."""
    people = usable(participants, windows, condition)
    pids = people["participant_id"].tolist()
    _, n_chans, n_times = windows[pids[0]][0].shape
    model = build_model(model_name, n_chans, n_times, seed)
    feats = participant_features(model, windows, pids, condition)
    prob, auc, bacc = probe(feats, people["label"].to_numpy(), cfg)
    return dict(prob=prob, auc=auc, bacc=bacc, participant_id=np.array(pids),
                label=people["label"].to_numpy())


def significant_digits(runs):
    """Significant decimal digits shared by runs (axis 0): -log10(std / |mean|).

    This is the usual Monte Carlo Arithmetic estimate. Values are capped at
    15.9 (float64 limit) when all runs agree exactly.
    """
    mean, std = runs.mean(axis=0), runs.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        digits = -np.log10(std / np.abs(mean))
    digits[std == 0] = 15.9
    return np.clip(np.nan_to_num(digits), 0, 15.9)


def summarize(prob, auc):
    """Summary numbers for a stack of runs: prob (n_runs, n_participants), auc (n_runs,)."""
    return dict(
        n_runs=len(auc),
        auc_mean=auc.mean(),
        auc_std=auc.std(ddof=1),
        auc_min=auc.min(),
        auc_max=auc.max(),
        prob_std_mean=prob.std(axis=0, ddof=1).mean(),  # avg per-participant spread
        prob_sig_digits_mean=significant_digits(prob).mean(),
    )
