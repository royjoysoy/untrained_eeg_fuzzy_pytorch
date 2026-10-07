"""Full automated EEG cleaning, the alternative to the minimal preprocessing in data.py.

Selected with `cleaning: full` in the config (use its own cache_dir). Per participant,
run-01, following current practice for resting EEG:

  0. keep only the 1-minute eyes-open / eyes-closed blocks, joined end to end. The rest
     of the recording (set-up, pauses) holds artifacts ~30x larger than the blocks and
     would dominate bad-channel detection and ICA.
  1. keep the 60 scalp channels, standard 10-05 positions
  2. remove 60 Hz line noise and harmonics (notch), recorded in the US
  3. find bad channels (PyPREP: flat, deviation, correlation, noise, RANSAC) and
     interpolate them
  4. average reference
  5. ICA (extended Infomax, Picard solver) on a 1-100 Hz copy, components labelled with
     ICLabel; remove those labelled eye blink, muscle, heart, line noise or channel noise
     with probability >= 0.8
  6. band-pass 1-40 Hz, resample, cut each block into the same 2 s windows as data.py
  7. autoreject (local): repair bad channels per window, drop windows it cannot repair

Returns the windows plus a small report (bad channels, removed components, dropped
windows) saved in the cache next to them.
"""

import numpy as np
import pandas as pd

ARTIFACTS = {"eye blink", "muscle artifact", "heart beat", "line noise", "channel noise"}
ICLABEL_MIN_PROBA = 0.8  # common choice; lower it to remove more components


def clean_subject(cfg, participant_id):
    import mne
    from autoreject import AutoReject
    from mne_icalabel import label_components
    from pyprep.find_noisy_channels import NoisyChannels

    from .data import _data_dir, find_blocks

    eeg_dir = _data_dir(cfg) / participant_id / "eeg"
    stem = f"{participant_id}_task-Rest_run-{cfg['run']}"
    raw = mne.io.read_raw_eeglab(eeg_dir / f"{stem}_eeg.set", preload=True, verbose="error")
    raw.pick(cfg["channels"])
    raw.set_montage("standard_1005", match_case=False, verbose="error")

    # 0. the complete blocks only, joined (same block choice as data.py)
    seg = cfg["segment_s"]
    events = pd.read_csv(eeg_dir / f"{stem}_events.tsv", sep="\t")
    blocks = [(t, c) for t, c in find_blocks(events, seg) if t + seg <= raw.times[-1]]
    pieces = [raw.copy().crop(t, t + seg, include_tmax=False) for t, _ in blocks]
    raw = mne.concatenate_raws(pieces, verbose="error")  # filters below stop at the joins

    # 2. line noise
    raw.notch_filter(np.arange(60, raw.info["sfreq"] / 2, 60), verbose="error")

    # 3. bad channels (PREP), then interpolation
    noisy = NoisyChannels(raw, random_state=0)
    noisy.find_all_bads(ransac=True)
    bads = noisy.get_bads()
    raw.info["bads"] = bads
    if bads:
        raw.interpolate_bads(reset_bads=True, verbose="error")

    # 4. average reference
    raw.set_eeg_reference("average", verbose="error")

    # 5. ICA + ICLabel. ICLabel was trained on 1-100 Hz, average-referenced data.
    ica_raw = raw.copy().filter(1.0, 100.0, verbose="error")
    rank = len(raw.ch_names) - len(bads) - 1  # interpolation and average reference lose rank
    ica = mne.preprocessing.ICA(n_components=rank, method="picard", random_state=0, max_iter="auto",
                                fit_params=dict(ortho=False, extended=True), verbose="error")
    ica.fit(ica_raw, verbose="error")
    ic = label_components(ica_raw, ica, method="iclabel")
    labels, proba = ic["labels"], ic["y_pred_proba"]
    # remove only confident artifact labels: the top label alone removes ~half the components
    exclude = [i for i, (label, p) in enumerate(zip(labels, proba)) if label in ARTIFACTS and p >= ICLABEL_MIN_PROBA]

    # 6. same band and rate as the minimal pipeline, then 2 s windows per block
    raw.filter(cfg["l_freq"], cfg["h_freq"], verbose="error")
    ica.apply(raw, exclude=exclude, verbose="error")
    raw.resample(cfg["sfreq"], verbose="error")
    data = raw.get_data()  # volts; blocks back to back
    n_seg, win = int(round(seg * cfg["sfreq"])), int(round(cfg["window_s"] * cfg["sfreq"]))
    n_win = n_seg // win
    X, cond = [], []
    for k, (_, condition) in enumerate(blocks):
        block = data[:, k * n_seg : k * n_seg + n_win * win]
        X.append(block.reshape(len(block), n_win, win).transpose(1, 0, 2))
        cond += [condition] * n_win
    X, cond = np.concatenate(X), np.array(cond)

    # 7. autoreject on the windows
    epochs = mne.EpochsArray(X, raw.info, verbose="error")
    ar = AutoReject(random_state=0, n_jobs=1, verbose=False)
    epochs_clean, log = ar.fit_transform(epochs, return_log=True)
    keep = ~log.bad_epochs
    X_clean = (epochs_clean.get_data() * 1e6).astype(np.float32)  # volts -> microvolts

    report = dict(
        bad_channels=np.array(bads, dtype=str),
        ica_excluded=np.array([labels[i] for i in exclude], dtype=str),
        n_ica=np.array(rank),
        n_windows_before=np.array(len(X)),
    )
    return X_clean, cond[keep], raw.ch_names, report
