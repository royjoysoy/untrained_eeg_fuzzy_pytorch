"""Probes for the optimization study, built on untrained_eeg.tasks (phase 1).

* `c1`      : exactly untrained_eeg.tasks.probe(..., variant="c1") -> same numbers as
              results/phase1/<dataset>/metrics.csv for the same features.
* `generic` : the same outer splits (tasks._outer_splits) and the same scoring, with any
              sklearn estimator; used for the Riemannian tangent space, which is fitted
              inside each training fold. With the c1 pipeline it equals `c1` (tested).
* non-finite feature values (e.g. Hjorth mobility of a flat channel, 0/0) are set to NaN
  and imputed with the TRAINING-fold median of that feature (SimpleImputer in the pipeline);
  feature sets without them go through exactly the c1 pipeline.
* `nested`  : selection among candidate feature sets inside each outer training set
              (inner splits of the same kind as the outer ones, as in tasks.py's nested C);
              outer test samples only score the selected candidate.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from untrained_eeg.tasks import MULTICLASS, WITHIN_SUBJECT, _outer_splits, auc_score, predicted, probe


def c1(X, y, groups, task, cfg):
    prob, auc, bacc = probe(X, y, groups, task, cfg["cv_seed"], cfg["cv_folds"])
    return prob, auc, bacc


class RiemannTangent(BaseEstimator, TransformerMixin):
    """Upper-triangular SPD column blocks -> tangent space at their training-fold Riemannian mean."""

    def __init__(self, spd_blocks=()):
        self.spd_blocks = spd_blocks

    @staticmethod
    def _mats(X, s, n):
        iu = np.triu_indices(n)
        M = np.zeros((len(X), n, n))
        M[:, iu[0], iu[1]] = X[:, s:s + len(iu[0])]
        M = M + np.triu(M, 1).transpose(0, 2, 1)
        return M + 1e-10 * np.trace(M, axis1=1, axis2=2)[:, None, None] * np.eye(n)  # numerical floor

    def fit(self, X, y=None):
        from pyriemann.tangentspace import TangentSpace
        self.ts_ = [TangentSpace(metric="riemann").fit(self._mats(X, s, n)) for s, n in self.spd_blocks]
        return self

    def transform(self, X):
        keep = np.ones(X.shape[1], bool)
        parts = []
        for ts, (s, n) in zip(self.ts_, self.spd_blocks):
            keep[s:s + n * (n + 1) // 2] = False
            parts.append(ts.transform(self._mats(X, s, n)))
        return np.hstack([X[:, keep]] + parts)


def estimator(spd_blocks=(), impute=False):
    """The c1 probe (standardize + LogisticRegression(max_iter=5000)); training-fold median
    imputation first if the features have non-finite values, then the tangent space if needed."""
    steps = [SimpleImputer(strategy="median", keep_empty_features=True)] if impute else []
    steps += [RiemannTangent(tuple(map(tuple, spd_blocks)))] if spd_blocks else []
    return make_pipeline(*steps, StandardScaler(), LogisticRegression(max_iter=5000))


def finite_or_nan(X):
    """(X with inf -> NaN, whether imputation is needed)."""
    bad = ~np.isfinite(X)
    if not bad.any():
        return X, False
    X = X.astype(np.float64, copy=True)
    X[bad] = np.nan
    return X, True


def _proba(est, X, y, tr, te):
    p = est.fit(X[tr], y[tr]).predict_proba(X[te])
    return p


def _finish(y, prob, task):
    if task not in MULTICLASS:
        prob = prob[:, 1]
    return prob, auc_score(y, prob), balanced_accuracy_score(y, predicted(prob))


def generic(X, y, groups, task, cfg, spd_blocks=()):
    X, imp = finite_or_nan(X)
    n_cls = len(np.unique(y))
    prob = np.zeros((len(y), n_cls))
    for tr, te in _outer_splits(X, y, groups, task, cfg["cv_seed"], cfg["cv_folds"]):
        prob[te] = _proba(estimator(spd_blocks, imp), X, y, tr, te)
    return _finish(y, prob, task)


def _inner_score(X, y, groups, task, cfg, spd_blocks):
    X, imp = finite_or_nan(X)
    n_cls = len(np.unique(y))
    prob = np.zeros((len(y), n_cls))
    kind = "" if task in WITHIN_SUBJECT else task
    for tr, te in _outer_splits(X, y, groups, kind, cfg["cv_seed"], cfg["cv_folds"]):
        prob[te] = _proba(estimator(spd_blocks, imp), X, y, tr, te)
    return auc_score(y, prob if task in MULTICLASS else prob[:, 1])


def nested(cands: dict, y, groups, task, cfg):
    """cands: {name: (X, spd_blocks)} over the same samples. Returns prob, auc, bacc, chosen list."""
    names = list(cands)
    any_X = cands[names[0]][0]
    prob = np.zeros((len(y), len(np.unique(y))))
    chosen = []
    for tr, te in _outer_splits(any_X, y, groups, task, cfg["cv_seed"], cfg["cv_folds"]):
        scores = [_inner_score(cands[n][0][tr], y[tr], groups[tr], task, cfg, cands[n][1]) for n in names]
        best = names[int(np.argmax(scores))]
        chosen.append(best)
        X, spd = cands[best]
        X, imp = finite_or_nan(X)
        prob[te] = _proba(estimator(spd, imp), X, y, tr, te)
    p, auc, bacc = _finish(y, prob, task)
    return p, auc, bacc, chosen
