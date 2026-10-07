"""Load ds003478, cut 1-minute eyes-open / eyes-closed blocks, cache as windows.

Pipeline for one participant (run-01 only):
  1. read the EEGLAB .set file with MNE
  2. keep the 60 scalp channels, average reference, band-pass, resample
  3. find each 1-minute block from the triggers and cut it into short windows
  4. save the windows to <cache_dir>/<participant>.npz

The cache is plain NumPy, so later steps (including the Fuzzy PyTorch container)
do not need MNE.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]

# Trigger codes 1-6 are sent every 500 ms during a 1-minute block.
# Odd codes = eyes closed, even codes = eyes open. Each code marks one block.
BLOCK_CODES = range(1, 7)


def load_config(path=REPO / "configs" / "ds003478.yaml"):
    """Read the YAML config and resolve paths."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["data_dir"] = os.environ.get("UNTRAINED_EEG_DATA") or cfg.get("data_dir")
    for key in ("cache_dir", "results_dir"):
        cfg[key] = REPO / cfg[key]  # relative to the repo root (absolute paths stay as-is)
    return cfg


def _data_dir(cfg):
    if not cfg["data_dir"]:
        raise SystemExit(
            "No data path. Run: export UNTRAINED_EEG_DATA=/path/to/ds003478\n"
            "(or set data_dir in the config). To download: scripts/00_download_data.sh"
        )
    return Path(cfg["data_dir"])


def load_participants(cfg):
    """Participants with a high or low BDI label, exclusions removed."""
    df = pd.read_csv(_data_dir(cfg) / "participants.tsv", sep="\t")
    df = df[~df["Original_ID"].isin(cfg["exclude_original_ids"])]
    df = df.dropna(subset=["BDI"])
    low = df["BDI"] <= cfg["bdi_low_max"]
    high = df["BDI"] >= cfg["bdi_high_min"]
    df = df[low | high].copy()
    df["label"] = (df["BDI"] >= cfg["bdi_high_min"]).astype(int)  # 1 = high BDI
    return df[["participant_id", "Original_ID", "BDI", "label"]].reset_index(drop=True)


def find_blocks(events, segment_s):
    """Return [(onset_s, 'open'|'closed'), ...], one per complete 1-minute block.

    For each code 1-6 we take its first uninterrupted run of 500 ms ticks.
    Blocks shorter than `segment_s` (recording started late, etc.) are skipped.
    """
    ev = events[events["value"].astype(str).isin([str(c) for c in BLOCK_CODES])]
    codes = ev["value"].astype(int).to_numpy()
    onsets = ev["onset"].to_numpy()
    blocks = []
    for code in BLOCK_CODES:
        idx = np.flatnonzero(codes == code)
        if len(idx) == 0:
            continue
        # keep only the first run of consecutive rows with this code
        breaks = np.flatnonzero(np.diff(idx) > 1)
        run = idx[: breaks[0] + 1] if len(breaks) else idx
        start, last = onsets[run[0]], onsets[run[-1]]
        if last + 0.5 - start >= segment_s:
            blocks.append((start, "closed" if code % 2 else "open"))
    return sorted(blocks)


def preprocess_subject(cfg, participant_id):
    """Return windows X (n_windows, n_channels, n_times) in microvolts and their conditions."""
    import mne  # imported here so the cache can be read without MNE

    eeg_dir = _data_dir(cfg) / participant_id / "eeg"
    stem = f"{participant_id}_task-Rest_run-{cfg['run']}"
    raw = mne.io.read_raw_eeglab(eeg_dir / f"{stem}_eeg.set", preload=True, verbose="error")
    raw.pick(cfg["channels"])  # same channels, same order, for everyone
    raw.set_eeg_reference("average", verbose="error")
    raw.filter(cfg["l_freq"], cfg["h_freq"], verbose="error")
    raw.resample(cfg["sfreq"], verbose="error")
    events = pd.read_csv(eeg_dir / f"{stem}_events.tsv", sep="\t")
    X, cond = cut_windows(cfg, raw.get_data() * 1e6, events)  # volts -> microvolts
    return X, cond, raw.ch_names


def cut_windows(cfg, data, events):
    """Cut (n_channels, n_samples) data at cfg['sfreq'] into the windows of each complete block."""
    sf, win = cfg["sfreq"], int(cfg["window_s"] * cfg["sfreq"])
    n_win = int(cfg["segment_s"] / cfg["window_s"])
    X, cond = [], []
    for onset, condition in find_blocks(events, cfg["segment_s"]):
        start = int(round(onset * sf))
        if start + n_win * win > data.shape[1]:
            continue  # block runs past the end of the recording
        seg = data[:, start : start + n_win * win]
        X.append(seg.reshape(len(seg), n_win, win).transpose(1, 0, 2))
        cond += [condition] * n_win
    return np.concatenate(X).astype(np.float32), np.array(cond)


def load_windows(cfg, participant_id, build=True):
    """Windows for one participant, read from the cache (built if missing)."""
    path = Path(cfg["cache_dir"]) / f"{participant_id}_run-{cfg['run']}.npz"
    if not path.exists():
        if not build:
            raise SystemExit(f"Missing cache {path}. Run scripts/01_build_cache.py first.")
        report = {}
        if cfg.get("cleaning", "minimal") == "full":
            from .clean import clean_subject  # needs pyprep, mne-icalabel, autoreject
            X, cond, ch_names, report = clean_subject(cfg, participant_id)
        else:
            X, cond, ch_names = preprocess_subject(cfg, participant_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, X=X, cond=cond, ch_names=ch_names, **report)
    f = np.load(path)
    return f["X"], f["cond"]


def load_dataset(cfg, build=True, verbose=True):
    """Participants table plus {participant_id: (X, cond)} for everyone."""
    participants = load_participants(cfg)
    if verbose:
        print(f"loading {len(participants)} participants from {cfg['cache_dir']} ...")
    windows = {pid: load_windows(cfg, pid, build=build) for pid in participants["participant_id"]}
    return participants, windows
