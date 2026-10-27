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
    # each dataset has its own variable (UNTRAINED_EEG_DATA for ds003478, the first one)
    cfg["data_dir"] = os.environ.get(cfg.get("data_env", "UNTRAINED_EEG_DATA")) or cfg.get("data_dir")
    for key in ("cache_dir", "results_dir"):
        cfg[key] = REPO / cfg[key]  # relative to the repo root (absolute paths stay as-is)
    return cfg


# Variants every dataset has: the sampling rates (and window) foundation models expect.
COMMON_VARIANTS = {"fm200": {"sfreq": 200}, "fm250": {"sfreq": 250},
                   "fm200_30s": {"sfreq": 200, "window_s": 30}}


def variant_config(cfg, name):
    """The config for one preprocessing variant (`preprocessing` in the config, or COMMON_VARIANTS).

    "standard" is the top-level settings. Other variants override some of them (e.g. the
    input a pretrained model was trained on) and get their own cache folder.
    """
    if name == "standard":
        return cfg
    out = {**cfg, **{**COMMON_VARIANTS, **cfg.get("preprocessing", {})}[name]}
    out["cache_dir"] = Path(cfg["cache_dir"]) / name
    return out


def _data_dir(cfg):
    """The BIDS dataset folder, downloaded first if it is not there yet.

    Default location: data/raw/<dataset> in the repo (git-ignored). UNTRAINED_EEG_DATA or
    data_dir in the config point to an existing copy instead (e.g. a shared /project folder).
    """
    if cfg["data_dir"]:
        return Path(cfg["data_dir"])  # an existing copy: never written to
    src = cfg["openneuro"]
    path = REPO / "data" / "raw" / src["id"]
    done = path / ".download_complete"  # written last, so an interrupted download resumes
    if not done.exists():
        from .download import download_openneuro

        print(f"downloading {src['id']} into {path}")
        if "urls" in src:  # plain files on a web server (BNCI)
            from .download import download_urls

            download_urls(src["urls"], path)
        elif "bucket" in src:  # another public S3 bucket (MPI-LEMON, PhysioNet's mirror)
            download_openneuro(src["prefix"], path, src["include"], bucket=src["bucket"])
        else:
            download_openneuro(src["id"], path, src["include"], src["version"])
        done.touch()
    return path


def load_participants(cfg):
    """Participants and their labels.

    ds003478: those with a high or low BDI label (column `label`), exclusions removed.
    Datasets with `tasks` in their config: everyone, with one column per task
    (1 = positive group, 0 = negative group, NaN = not in that task).
    """
    if "tasks" in cfg:
        return _task_participants(cfg)
    df = pd.read_csv(_data_dir(cfg) / "participants.tsv", sep="\t")
    df = df[~df["Original_ID"].isin(cfg["exclude_original_ids"])]
    df = df.dropna(subset=["BDI"])
    low = df["BDI"] <= cfg["bdi_low_max"]
    high = df["BDI"] >= cfg["bdi_high_min"]
    df = df[low | high].copy()
    df["label"] = (df["BDI"] >= cfg["bdi_high_min"]).astype(int)  # 1 = high BDI
    return df[["participant_id", "Original_ID", "BDI", "label"]].reset_index(drop=True)


def _task_participants(cfg):
    if "participants" in cfg:  # a fixed list (BNCI has no participants.tsv)
        df = pd.DataFrame({"participant_id": cfg["participants"]})
        return df.assign(**{task: np.nan for task in cfg["tasks"]})  # labels come from trials
    if "participants_file" in cfg:  # e.g. LEMON's META table: id in the first column
        df = pd.read_csv(_data_dir(cfg) / cfg["participants_file"])
        df = df.rename(columns={df.columns[0]: "participant_id"})
        have = {p.name for p in (_data_dir(cfg) / cfg["recordings_dir"]).iterdir()}
        df = df[df["participant_id"].isin(have)]  # those with EEG
    else:
        df = pd.read_csv(_data_dir(cfg) / "participants.tsv", sep="\t")
    df["group"] = df["participant_id"].str.extract(r"sub-([a-z]+)", expand=False)  # e.g. pd, hc
    for task, t in cfg["tasks"].items():
        df[task] = np.where(df[t["column"]].isin(t["positive"]), 1.0,
                            np.where(df[t["column"]].isin(t["negative"]), 0.0, np.nan))
    return df[["participant_id", *cfg["tasks"]]].reset_index(drop=True)


def _clean(raw, cfg):
    """Shared preprocessing of a loaded MNE recording: (n_channels, n_samples) in microvolts.

    Channels, average reference, a variant's channel subset (`pick`), band-pass, resample,
    microvolts, and a variant's exponential moving standardization (`ems`).
    """
    raw.pick(cfg["channels"])
    if cfg.get("reference", "average") == "average":  # "none" for bipolar derivations (sleep)
        raw.set_eeg_reference("average", verbose="error")
    if cfg.get("pick"):
        raw.pick(cfg["pick"])
    raw.filter(cfg["l_freq"], cfg["h_freq"], verbose="error")
    raw.resample(cfg["sfreq"], verbose="error")
    data = raw.get_data() * 1e6
    if cfg.get("ems"):
        from braindecode.preprocessing import exponential_moving_standardize

        data = exponential_moving_standardize(data, **cfg["ems"])
    return data, raw.ch_names


def preprocess_bnci(cfg, participant_id):
    """BNCI 2014-001 (BCI IV 2a): left and right hand trials of both sessions.

    Each run is cleaned as one recording, then every trial gives the windows of the
    imagery period (`trial_window`, seconds after the trial start; the cue is at 2 s).
    `cond` is the trial's class ("left", "right"); feet and tongue trials are dropped.
    """
    import mne
    import scipy.io

    win = int(cfg["window_s"] * cfg["sfreq"])
    t0, t1 = cfg["trial_window"]
    X, cond = [], []
    for session in "TE":
        mat = scipy.io.loadmat(_data_dir(cfg) / f"{participant_id}{session}.mat",
                               squeeze_me=True, struct_as_record=False)
        for run in mat["data"]:
            if not np.size(run.trial):
                continue  # the first runs of a session have no trials (EOG calibration)
            info = mne.create_info(cfg["all_channels"], run.fs, "eeg")
            raw = mne.io.RawArray(run.X.T * 1e-6, info, verbose="error")  # stored in microvolts
            data, ch_names = _clean(raw, cfg)
            scale = cfg["sfreq"] / run.fs
            for onset, y in zip(np.atleast_1d(run.trial), np.atleast_1d(run.y)):
                name = {1: "left", 2: "right"}.get(int(y))
                if name is None:
                    continue
                start = int(round((onset - 1) * scale + t0 * cfg["sfreq"]))  # onset: 1-based sample
                n = int((t1 - t0) / cfg["window_s"])
                seg = data[:, start : start + n * win]
                X.append(seg.reshape(len(seg), n, win).transpose(1, 0, 2))
                cond += [name] * n
    return np.concatenate(X).astype(np.float32), np.array(cond), ch_names


SLEEP_STAGES = {"Sleep stage W": "W", "Sleep stage 1": "N1", "Sleep stage 2": "N2",
                "Sleep stage 3": "N3", "Sleep stage 4": "N3", "Sleep stage R": "REM"}  # R&K -> AASM


def preprocess_sleep(cfg, participant_id):
    """Sleep-EDF sleep cassette: 30 s epochs of both nights of one subject, labelled by stage.

    Stages 3 and 4 merge into N3; movement and unscored epochs are dropped. Wake is kept only
    within `wake_margin_min` of sleep (the long wake periods before and after are trimmed).
    Then at most `max_per_class` epochs per stage and night are drawn (fixed seed), so every
    night weighs about the same and the cache stays small.
    """
    import glob

    import mne

    subject = participant_id.removeprefix("sub-")
    folder = _data_dir(cfg) / "sleep-cassette"
    X, cond = [], []
    for night in "12":
        psg = sorted(glob.glob(str(folder / f"SC4{subject}{night}E?-PSG.edf")))
        hyp = sorted(glob.glob(str(folder / f"SC4{subject}{night}E?-Hypnogram.edf")))
        if not psg or not hyp:
            continue  # a few subjects have one night only
        raw = mne.io.read_raw_edf(psg[0], include=cfg["channels"], preload=True, verbose="error")
        data, ch_names = _clean(raw, cfg)
        sf, win = cfg["sfreq"], int(cfg["window_s"] * cfg["sfreq"])
        ann = mne.read_annotations(hyp[0])
        epochs = [(o + 30 * k, SLEEP_STAGES[d]) for o, dur, d in zip(ann.onset, ann.duration, ann.description)
                  if d in SLEEP_STAGES for k in range(int(dur // 30))]
        sleep = [o for o, s in epochs if s != "W"]
        lo, hi = min(sleep) - 60 * cfg["wake_margin_min"], max(sleep) + 60 * cfg["wake_margin_min"]
        epochs = [(o, s) for o, s in epochs if lo <= o <= hi and int(o * sf) + win <= data.shape[1]]
        rng = np.random.default_rng(int(subject) * 10 + int(night))
        for stage in ("W", "N1", "N2", "N3", "REM"):
            onsets = [o for o, s in epochs if s == stage]
            take = rng.permutation(len(onsets))[: cfg["max_per_class"]]
            for i in sorted(take):
                start = int(onsets[i] * sf)
                X.append(data[:, start : start + win])
                cond.append(stage)
    return np.stack(X).astype(np.float32), np.array(cond), ch_names


def recording_path(cfg, participant_id):
    """The participant's EEG file, from the config's `recording` template."""
    session = None
    if "sessions" in cfg:
        session = cfg["sessions"][participant_id.removeprefix("sub-").rstrip("0123456789")]
    return _data_dir(cfg) / cfg["recording"].format(pid=participant_id, session=session)


def preprocess_resting(cfg, participant_id):
    """Windows of a continuous resting recording, grouped in `segment_s` segments.

    Same steps as ds003478: channels, average reference, band-pass, resample, microvolts.
    Then every complete segment is cut into windows; a partial last segment is dropped.
    """
    import mne

    path = recording_path(cfg, participant_id)
    read = mne.io.read_raw_eeglab if path.suffix == ".set" else mne.io.read_raw_bdf
    data, ch_names = _clean(read(path, preload=True, verbose="error"), cfg)
    X = _segments(data, cfg)
    return X, np.array([cfg["condition"]] * len(X)), ch_names


def _segments(data, cfg):
    """Windows of every complete `segment_s` segment: (n_windows, n_channels, n_times)."""
    win = int(cfg["window_s"] * cfg["sfreq"])
    n_win = int(cfg["segment_s"] / cfg["window_s"])
    n_seg = data.shape[1] // (n_win * win)
    seg = data[:, : n_seg * n_win * win]
    return seg.reshape(len(seg), n_seg * n_win, win).transpose(1, 0, 2).astype(np.float32)


def preprocess_lemon(cfg, participant_id):
    """MPI-LEMON preprocessed eyes-closed and eyes-open recordings, cut like resting ones.

    The authors removed bad channels per recording (57-61 left); the missing ones are
    rebuilt by spherical-spline interpolation, so everyone has the same `channels`.
    """
    import mne

    X, cond = [], []
    for condition, tag in (("closed", "EC"), ("open", "EO")):
        path = _data_dir(cfg) / cfg["recording"].format(pid=participant_id, tag=tag)
        raw = mne.io.read_raw_eeglab(path, preload=True, verbose="error")
        missing = [c for c in cfg["channels"] if c not in raw.ch_names]
        if missing:
            zeros = mne.io.RawArray(np.zeros((len(missing), raw.n_times)),
                                    mne.create_info(missing, raw.info["sfreq"], "eeg"), verbose="error")
            raw.add_channels([zeros], force_update_info=True)
            raw.set_montage("standard_1005", verbose="error")
            raw.info["bads"] = missing
            raw.interpolate_bads(reset_bads=True, verbose="error")
        data, ch_names = _clean(raw, cfg)
        w = _segments(data, cfg)
        X.append(w)
        cond += [condition] * len(w)
    return np.concatenate(X), np.array(cond), ch_names


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
    if cfg.get("pick"):  # a variant's subset, taken after the 60-channel average reference
        raw.pick([c.upper() for c in cfg["pick"]])
    raw.filter(cfg["l_freq"], cfg["h_freq"], verbose="error")
    raw.resample(cfg["sfreq"], verbose="error")
    data = raw.get_data() * 1e6  # volts -> microvolts
    if cfg.get("ems"):  # as in braindecode's tutorials, on the continuous recording
        from braindecode.preprocessing import exponential_moving_standardize

        data = exponential_moving_standardize(data, **cfg["ems"])

    events = pd.read_csv(eeg_dir / f"{stem}_events.tsv", sep="\t")
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
    return np.concatenate(X).astype(np.float32), np.array(cond), raw.ch_names


def load_windows(cfg, participant_id, build=True):
    """Windows for one participant, read from the cache (built if missing)."""
    name = f"{participant_id}_run-{cfg['run']}" if "run" in cfg else participant_id
    path = Path(cfg["cache_dir"]) / f"{name}.npz"
    if not path.exists():
        if not build:
            raise SystemExit(f"Missing cache {path}. Run scripts/01_run_untrained.py first.")
        prep = {"resting": preprocess_resting, "bnci": preprocess_bnci,
                "lemon": preprocess_lemon, "sleep": preprocess_sleep}.get(cfg.get("loader"), preprocess_subject)
        X, cond, ch_names = prep(cfg, participant_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, X=X, cond=cond, ch_names=ch_names)
    f = np.load(path)
    return f["X"], f["cond"]


def load_dataset(cfg, build=True, verbose=True):
    """Participants table plus {participant_id: (X, cond)} for everyone."""
    participants = load_participants(cfg)
    if verbose:
        print(f"loading {len(participants)} participants from {cfg['cache_dir']} ...")
    windows = {pid: load_windows(cfg, pid, build=build) for pid in participants["participant_id"]}
    return participants, windows
