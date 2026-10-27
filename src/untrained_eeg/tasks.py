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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
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


def probe(X, y, groups, task, cv_seed, folds):
    """Out-of-fold P(class 1), AUC and balanced accuracy."""
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
