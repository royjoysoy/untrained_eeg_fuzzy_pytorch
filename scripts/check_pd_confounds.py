"""Is EEGPT's Parkinson's result (ds002778, AUC 0.85) amplitude, recording, or a few people?

  A  trivial features with the same probe and folds: per-channel log variance, per-channel
     log band power (delta, theta, alpha, beta, gamma), and one number per participant (log
     RMS over all channels); AUC over 30 CV splits and a refit permutation p (split 0)
  C  recording metadata by group: duration and channel count of the raw file, number of
     segments, RMS amplitude after preprocessing (AUC of each as a patient-vs-control score)
  D  EEGPT: AUC with each participant left out (mean over 10 CV splits), and how EEGPT's
     out-of-fold score relates to log RMS

Reads the preprocessed cache (data/cache/ds002778), the raw BDF headers and
results/phase1/ds002778/features/eegpt. Prints a report; writes nothing.

    python scripts/check_pd_confounds.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.data import load_config, load_dataset, recording_path
from untrained_eeg.tasks import probe, task_data, task_labels

TASK = "pd_vs_control"
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40)}
SPLITS = range(30)
N_PERM = 1000

cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "ds002778.yaml")
labels = task_labels(cfg)
participants, windows = load_dataset(cfg, build=False, verbose=False)


def segment_features(X):
    """Window features (log variance, log band power) of one participant, in microvolts."""
    logvar = np.log(X.var(axis=2))  # windows x channels
    f, psd = welch(X, fs=cfg["sfreq"], nperseg=X.shape[2], axis=2)
    bands = np.stack([np.log(psd[:, :, (f >= lo) & (f < hi)].mean(axis=2)) for lo, hi in BANDS.values()], axis=2)
    return logvar, bands.reshape(len(X), -1)


def evaluate(name, feats, pid, cond):
    X, y, groups, _ = task_data(feats, cond, pid, TASK, labels)
    aucs = [probe(X, y, groups, TASK, s, cfg["cv_folds"])[1] for s in SPLITS]
    obs = aucs[0]
    rng = np.random.default_rng(0)
    null = [probe(X, rng.permutation(y), groups, TASK, 0, cfg["cv_folds"])[1] for _ in range(N_PERM)]
    p = (1 + np.sum(np.array(null) >= obs)) / (1 + N_PERM)
    print(f"  {name:28s} {X.shape[1]:4d} feat  AUC {obs:.3f}  30 splits {np.mean(aucs):.3f} "
          f"± {np.std(aucs):.3f}  null 95% {np.quantile(null, 0.95):.3f}  p {p:.4f}")
    return X, y


# Segment-level rows, as the phase 1 features: windows averaged within each 60 s segment
rows = {"logvar": [], "band": [], "rms": []}
pid, cond, meta = [], [], []
for p_id, (X, c) in windows.items():
    logvar, band = segment_features(X)
    n_seg = len(X) // 30
    for s in range(n_seg):
        w = slice(30 * s, 30 * (s + 1))
        rows["logvar"].append(logvar[w].mean(axis=0))
        rows["band"].append(band[w].mean(axis=0))
        rows["rms"].append([np.log(np.sqrt((X[w] ** 2).mean()))])
        pid.append(p_id)
        cond.append(c[30 * s])
    meta.append(dict(participant_id=p_id, segments=n_seg, rms_uv=float(np.sqrt((X ** 2).mean()))))
pid, cond = np.array(pid), np.array(cond)

print("A. Trivial features, same probe and folds (EEGPT: AUC 0.846, 30 splits 0.834 ± 0.048)")
for name, key in [("log RMS, all channels", "rms"), ("log variance per channel", "logvar"),
                  ("log band power per channel", "band")]:
    evaluate(name, np.array(rows[key]), pid, cond)

print("\nC. Recording metadata, patients vs controls (AUC > 0.5: larger in patients)")
meta = pd.DataFrame(meta).set_index("participant_id")
for p_id in meta.index:
    import mne
    raw = mne.io.read_raw_bdf(recording_path(cfg, p_id), preload=False, verbose="error")
    meta.loc[p_id, "duration_s"] = raw.n_times / raw.info["sfreq"]
    meta.loc[p_id, "n_channels"] = len(raw.ch_names)
    meta.loc[p_id, "sfreq"] = raw.info["sfreq"]
meta["patient"] = labels.loc[meta.index, TASK].astype(int)
for col in ["duration_s", "n_channels", "sfreq", "segments", "rms_uv"]:
    g = meta.groupby("patient")[col]
    auc = roc_auc_score(meta.patient, meta[col]) if meta[col].nunique() > 1 else float("nan")
    print(f"  {col:12s} controls {g.median()[0]:8.1f}  patients {g.median()[1]:8.1f}  (medians)  AUC {auc:.3f}")

print("\nD. EEGPT: influence of single participants")
f = np.load(Path(cfg["results_dir"]) / "phase1" / "ds002778" / "features" / "eegpt" / "pretrained.npz")
X, y, groups, people = task_data(f["feats"], f["cond"], f["participant_id"], TASK, labels)
prob = probe(X, y, groups, TASK, 0, cfg["cv_folds"])[0]
loo = []
for i in range(len(y)):
    keep = np.arange(len(y)) != i
    loo.append(np.mean([probe(X[keep], y[keep], None, TASK, s, cfg["cv_folds"])[1] for s in range(10)]))
loo = pd.Series(loo, index=people)
print(f"  AUC without one participant (mean of 10 splits): {loo.min():.3f} to {loo.max():.3f}")
print(f"  lowest when leaving out: {', '.join(f'{p} ({v:.3f})' for p, v in loo.nsmallest(3).items())}")
oof = pd.Series(prob, index=people)
wrong = [p for p in people if (oof[p] > 0.5) != bool(labels.loc[p, TASK])]
print(f"  misclassified at P > 0.5 (split 0): {len(wrong)} of {len(y)}: {', '.join(wrong)}")
rho, p_rho = spearmanr(oof, np.log(meta.loc[people, "rms_uv"]))
print(f"  EEGPT out-of-fold score vs log RMS: Spearman {rho:.2f} (p {p_rho:.3f})")
for grp, g in meta.loc[people].groupby("patient"):
    r, pr = spearmanr(oof[g.index], np.log(g.rms_uv))
    print(f"    within {'patients' if grp else 'controls'}: Spearman {r:.2f} (p {pr:.3f})")
