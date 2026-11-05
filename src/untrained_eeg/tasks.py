"""The tasks of each dataset and their probe, shared by phase 1 (single models) and phase 2.

Same probe everywhere: standardize + logistic regression, stratified 5-fold CV.
ds003478:
  ec_eo        eyes closed vs open, one sample per 1-minute block; folds grouped by
               participant, so no participant is in both training and test
  bdi_closed   high vs low BDI, one sample per participant (mean of its eyes-closed blocks)
  bdi_open     the same with eyes-open blocks
Other datasets: the `tasks` of their config, one sample per participant (mean of all its
segments, or of one condition's for a task named <name>_closed / <name>_open), label 1 =
patient (or older) group, 0 = control (or younger).
mi_left_right  BNCI 2014-001 left vs right hand imagery, one sample per trial; each subject
               gets its own stratified 5-fold CV (within subject), AUC over all trials
sleep_stage    Sleep-EDF, 5 stages, one sample per 30 s epoch, folds grouped by subject;
               macro one-vs-rest AUC
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_participants

WITHIN_SUBJECT = {"mi_left_right"}  # trial-level tasks probed separately in each subject
MULTICLASS = {"sleep_stage": ["W", "N1", "N2", "N3", "REM"]}  # task: classes, in label order


def auc_score(y, prob):
    """AUC; macro one-vs-rest when prob has one column per class."""
    if prob.ndim == 2:
        return roc_auc_score(y, prob, multi_class="ovr", average="macro", labels=range(prob.shape[1]))
    return roc_auc_score(y, prob)


def subject_auc(y, prob, groups):
    """Mean and SD over subjects of the per-subject AUC (within-subject tasks, where each subject
    has its own probe: pooling their probabilities into one AUC mixes calibrations)."""
    aucs = [roc_auc_score(y[groups == g], prob[groups == g]) for g in np.unique(groups)]
    return np.mean(aucs), np.std(aucs, ddof=1)


def predicted(prob):
    """Predicted classes: the most probable one, or P > 0.5 for two classes."""
    return prob.argmax(axis=1) if prob.ndim == 2 else (prob > 0.5).astype(int)


TASKS = ("ec_eo", "bdi_closed", "bdi_open")  # ds003478


def task_names(cfg):
    return list(cfg["tasks"]) if "tasks" in cfg else list(TASKS)


def task_labels(cfg):
    """Participants table indexed by id: `label` (ds003478) or one column per task."""
    return load_participants(cfg).set_index("participant_id")


def task_data(feats, cond, pid, task, labels):
    """(X, y, groups, sample ids) for one task from block features (one row per block or segment)."""
    if task == "ec_eo":
        keep = np.isin(pid, labels.index)  # the same 113 participants as the BDI tasks
        block = pd.Series(pid[keep]).groupby(pid[keep]).cumcount().to_numpy()
        ids = [f"{p}/block{b}" for p, b in zip(pid[keep], block)]
        return feats[keep], (cond[keep] == "closed").astype(int), pid[keep], ids
    if task in MULTICLASS:  # one row per epoch, class in cond
        classes = MULTICLASS[task]
        keep = np.isin(cond, classes)
        epoch = pd.Series(pid[keep]).groupby(pid[keep]).cumcount().to_numpy()
        ids = [f"{p}/epoch{k}" for p, k in zip(pid[keep], epoch)]
        y = np.array([classes.index(c) for c in cond[keep]])
        return feats[keep], y, pid[keep], ids
    if task in WITHIN_SUBJECT:  # one row per trial, class in cond
        keep = np.isin(cond, ["left", "right"])
        trial = pd.Series(pid[keep]).groupby(pid[keep]).cumcount().to_numpy()
        ids = [f"{p}/trial{k}" for p, k in zip(pid[keep], trial)]
        return feats[keep], (cond[keep] == "right").astype(int), pid[keep], ids
    if task.startswith("bdi_"):
        condition = task.removeprefix("bdi_")
        people = [p for p in labels.index if ((pid == p) & (cond == condition)).any()]
        X = np.stack([feats[(pid == p) & (cond == condition)].mean(axis=0) for p in people])
        return X, labels.loc[people, "label"].to_numpy(), np.array(people), people
    rows = np.ones(len(pid), bool)  # "<task>_closed" / "<task>_open": one condition only
    if task.endswith(("_closed", "_open")):
        rows = cond == task.rpartition("_")[2]
    people = [p for p in labels.index if not np.isnan(labels.loc[p, task]) and ((pid == p) & rows).any()]
    X = np.stack([feats[(pid == p) & rows].mean(axis=0) for p in people])
    return X, labels.loc[people, task].to_numpy().astype(int), np.array(people), people


PROBES = ("c1", "nested", "pca")  # c1: the main probe; the others are a sensitivity analysis
NESTED_C = np.logspace(-4, 2, 7)  # nested: C chosen by inner CV (scikit-learn's C = 1 / L2 strength)
PCA_DIM = 32  # pca: components kept (at most the training fold size - 1), whitened, then C = 1


def _outer_splits(X, y, groups, task, cv_seed, folds):
    """The probe's CV splits: (train, test) index arrays, as in the main probe."""
    if task in WITHIN_SUBJECT:  # a CV per subject
        for g in np.unique(groups):
            i = np.flatnonzero(groups == g)
            for tr, te in StratifiedKFold(folds, shuffle=True, random_state=cv_seed).split(X[i], y[i]):
                yield i[tr], i[te]
    elif task in MULTICLASS or task == "ec_eo":  # samples of a participant in one fold
        yield from StratifiedGroupKFold(folds, shuffle=True, random_state=cv_seed).split(X, y, groups)
    else:
        yield from StratifiedKFold(folds, shuffle=True, random_state=cv_seed).split(X, y)


def _fit_predict(X, y, groups, tr, te, task, variant, cv_seed, folds, info):
    """Fit the sensitivity probe `variant` on the training fold, P(classes) on the test fold."""
    if variant == "pca":  # the same number of features for every extractor
        k = min(PCA_DIM, len(tr) - 1, X.shape[1])
        clf = make_pipeline(StandardScaler(), PCA(k, whiten=True, random_state=0), LogisticRegression(max_iter=5000))
        return clf.fit(X[tr], y[tr]).predict_proba(X[te])
    # nested: C chosen by an inner CV of the same kind (grouped where the outer one is), inside the
    # training fold only; within-subject tasks are already one subject per outer split
    inner = list(_outer_splits(X[tr], y[tr], groups[tr], "" if task in WITHIN_SUBJECT else task, cv_seed, folds))
    search = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                          {"logisticregression__C": NESTED_C}, cv=inner,
                          scoring="roc_auc_ovr" if task in MULTICLASS else "roc_auc")
    search.fit(X[tr], y[tr])
    if info is not None:
        info.append(search.best_params_["logisticregression__C"])
    return search.predict_proba(X[te])


def probe(X, y, groups, task, cv_seed, folds, variant="c1", info=None):
    """Out-of-fold P(class 1), AUC and balanced accuracy.

    variant: "c1" (the main probe: standardize + logistic regression, C = 1), "nested" (C chosen
    in each training fold by an inner CV over NESTED_C; chosen C appended to the list `info`) or
    "pca" (standardize, PCA to PCA_DIM whitened components, C = 1), both fitted on training folds
    only and with the same outer splits as the main probe.
    """
    if variant != "c1":
        prob = np.empty((len(y), len(np.unique(y))))
        for tr, te in _outer_splits(X, y, groups, task, cv_seed, folds):
            prob[te] = _fit_predict(X, y, groups, tr, te, task, variant, cv_seed, folds, info)
        if task not in MULTICLASS:
            prob = prob[:, 1]
        return prob, auc_score(y, prob), balanced_accuracy_score(y, predicted(prob))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    if task in WITHIN_SUBJECT:  # a probe per subject, out-of-fold P kept in the input order
        prob = np.empty(len(y))
        for g in np.unique(groups):
            i = np.flatnonzero(groups == g)
            cv = StratifiedKFold(folds, shuffle=True, random_state=cv_seed)
            prob[i] = cross_val_predict(clf, X[i], y[i], cv=cv, method="predict_proba")[:, 1]
        return prob, roc_auc_score(y, prob), balanced_accuracy_score(y, prob > 0.5)
    if task in MULTICLASS:  # epochs: keep each subject's epochs in one fold
        cv = StratifiedGroupKFold(folds, shuffle=True, random_state=cv_seed)
        prob = cross_val_predict(clf, X, y, groups=groups, cv=cv, method="predict_proba")
        return prob, auc_score(y, prob), balanced_accuracy_score(y, prob.argmax(axis=1))
    if task == "ec_eo":  # blocks: keep each participant's blocks in one fold
        cv = StratifiedGroupKFold(folds, shuffle=True, random_state=cv_seed)
    else:  # one sample per participant: no groups needed
        cv, groups = StratifiedKFold(folds, shuffle=True, random_state=cv_seed), None
    prob = cross_val_predict(clf, X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
    return prob, roc_auc_score(y, prob), balanced_accuracy_score(y, prob > 0.5)
