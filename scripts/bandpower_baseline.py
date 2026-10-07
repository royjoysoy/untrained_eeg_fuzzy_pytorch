"""Group C (#32, task 10): band-power baseline with the same probe as the random CNNs.

From the same cached 2 s windows: the power spectrum of each window (one Welch segment of
2 s, 0.5 Hz resolution), averaged over each participant's windows of one condition, then the
mean power per channel in delta (1-4 Hz), theta (4-8), alpha (8-13) and beta (13-30),
log10-transformed: 60 channels x 4 bands = 240 features. Same probe (standardize + logistic
regression), same stratified 5-fold split, same metrics, plus a label-permutation p-value.

Writes <out>/bandpower_features.npz, <out>/predictions.csv.gz, <out>/runs.csv.gz (tidy
format of probe_runs.py, model = "bandpower").

    python scripts/bandpower_baseline.py --config configs/paper.yaml --out results/paper/C10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.data import load_config, load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from permutation_null import permutation_test  # noqa: E402
from probe_runs import metrics, probe  # noqa: E402

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}


def band_features(X, sfreq):
    """log10 mean band power per channel and band, from windows X (n_windows, n_chans, n_times)."""
    freqs, psd = welch(X, fs=sfreq, nperseg=X.shape[-1], axis=-1)  # (n_windows, n_chans, n_freqs)
    psd = psd.mean(axis=0)  # average spectrum over windows
    return np.concatenate([np.log10(psd[:, (freqs >= lo) & (freqs < hi)].mean(axis=1)) for lo, hi in BANDS.values()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    participants, windows = load_dataset(cfg, build=False, verbose=False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    feats_out, pred_rows, run_rows = {}, [], []
    for condition in cfg["conditions"]:
        pids, X = [], []
        for pid in participants["participant_id"]:
            x, c = windows[pid]
            if (c == condition).any():
                pids.append(pid)
                X.append(band_features(x[c == condition], cfg["sfreq"]))
        X = np.stack(X)
        y = participants.set_index("participant_id").loc[pids, "label"].to_numpy()
        feats_out[condition] = X
        feats_out[f"{condition}_participant_id"] = np.array(pids)
        prob = probe(X, y, cfg["cv_seed"], cfg["cv_folds"])
        obs, null, p = permutation_test(X, y, cfg["cv_seed"], cfg["cv_folds"], args.n_perm, args.workers)
        key = dict(model="bandpower", source="baseline", weight_seed=-1, sr_seed=-1, condition=condition,
                   cv_seed=cfg["cv_seed"])
        run_rows.append(dict(key, n=len(y), **metrics(y, prob), perm_p=p, null_q95=np.quantile(null, 0.95)))
        pred_rows.append(pd.DataFrame(dict(key, participant_id=pids, label=y, prob=prob)))
        print(f"bandpower {condition:6s}: AUC {obs:.3f}, balanced accuracy {run_rows[-1]['balanced_accuracy']:.3f}, "
              f"permutation p = {p:.3f}", flush=True)
    names = [f"{b}_{ch}" for b in BANDS for ch in cfg["channels"]]
    np.savez(out / "bandpower_features.npz", feature_names=names, **feats_out)
    pd.concat(pred_rows).to_csv(out / "predictions.csv.gz", index=False, float_format="%.17g")
    pd.DataFrame(run_rows).to_csv(out / "runs.csv.gz", index=False, float_format="%.10g")
    print(f"wrote {out}")
