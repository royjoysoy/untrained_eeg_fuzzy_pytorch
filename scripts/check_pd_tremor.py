"""Does a model's Parkinson's score on ds003490 follow hand tremor rather than brain activity?

ds003490 recorded a 3-axis accelerometer on each participant's most tremor-affected hand.
For the two resting minutes used by the probe, this computes the log power of the
acceleration magnitude in the rest-tremor band (3-8 Hz), its AUC as a patient-vs-control
score, and its Spearman correlation with each model's out-of-fold score (phase 1, split 0).

    python scripts/check_pd_tremor.py --extractors eegpt shallow scattering
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.data import _data_dir, load_config, load_participants

TASK = "pd_vs_control"
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--extractors", nargs="+", default=["eegpt", "shallow", "scattering", "labram"])
args = parser.parse_args()

import mne  # noqa: E402

cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "ds003490.yaml")
meta = pd.read_csv(_data_dir(cfg) / "participants.tsv", sep="\t").set_index("participant_id")
labels = load_participants(cfg).set_index("participant_id")[TASK]
tremor = {}
for pid in labels.index:
    session = "02" if meta.loc[pid, "sess2_Med"] == cfg["session_med"] else "01"
    path = _data_dir(cfg) / cfg["recording"].format(pid=pid, session=session)
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose="error").pick(["X", "Y", "Z"])
    events = pd.read_csv(str(path).replace("_eeg.set", "_events.tsv"), sep="\t")
    sf, powers = raw.info["sfreq"], []
    for codes in cfg["rest_spans"].values():
        onsets = events.loc[events["value"].astype(str).str.strip().isin(codes), "onset"]
        if onsets.empty:
            continue
        a = raw.get_data(start=int(onsets.min() * sf), stop=int((onsets.min() + cfg["segment_s"]) * sf))
        mag = np.linalg.norm(a - a.mean(axis=1, keepdims=True), axis=0)
        f, psd = welch(mag, fs=sf, nperseg=int(4 * sf))
        powers.append(psd[(f >= 3) & (f <= 8)].mean())
    tremor[pid] = np.log(np.mean(powers))
tremor = pd.Series(tremor)
y = labels.loc[tremor.index].astype(int)
print(f"accelerometer 3-8 Hz log power: patients {tremor[y == 1].median():.2f}, controls "
      f"{tremor[y == 0].median():.2f} (medians), AUC {roc_auc_score(y, tremor):.3f}")

pred = pd.read_csv(Path(cfg["results_dir"]) / "phase1" / "ds003490" / "predictions.csv.gz")
pred = pred[(pred.task == TASK) & (pred.cv_seed == 0)]
for ext in args.extractors:
    g = pred[pred.extractor == ext]
    if g.empty:
        continue
    score = g.groupby("sample").prob.mean()  # mean over instances (= ensemble score)
    common = score.index.intersection(tremor.index)
    rho, p = spearmanr(score[common], tremor[common])
    within = [spearmanr(score[common][y[common] == k], tremor[common][y[common] == k])[0] for k in (1, 0)]
    print(f"  {ext:12s} AUC {roc_auc_score(y[common], score[common]):.3f}  score vs tremor: Spearman "
          f"{rho:.2f} (p {p:.3f}); within patients {within[0]:.2f}, within controls {within[1]:.2f}")
