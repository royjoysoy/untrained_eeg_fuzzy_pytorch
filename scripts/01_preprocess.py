#!/usr/bin/env python
"""Minimal preprocessing of OpenNeuro ds003478 (resting EEG, run-01).

Per subject:
  1. Load the EEGLAB .set (66 channels, 500 Hz, Neuroscan Synamps2).
  2. Select canonical 60 or 64 electrodes; always exclude HEOG and VEOG.
  3. Band-pass 1-40 Hz, average reference, resample to 250 Hz.
  4. Cut the six 1-minute blocks using events.tsv:
       block codes 1, 3, 5 = eyes closed;  2, 4, 6 = eyes open
     (each block's 500 ms markers carry codes 1-6, 2 s markers 11-16).
  5. Save data/derivatives/<sub>.npz with
       X      (n_blocks, selected_electrodes, 15000) float32, in volts
       block  (n_blocks,)  int   block code 1-6
       cond   (n_blocks,)  str   "EO" / "EC"
       ch_names, sfreq

Excludes sub-038 (Original_ID 544, flagged INVALID PARTICIPANT in
participants.tsv). No ICA: eye artefacts remain, deliberately, to keep the
pipeline short and deterministic. Add ICA here if the probe needs it.

Usage:
  python scripts/01_preprocess.py --bids-root /path/to/ds003478 --out data/derivatives
"""

from __future__ import annotations

import argparse
import logging
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd

DROP = ["M1", "M2", "CB1", "CB2", "HEOG", "VEOG"]
CHANNELS_64 = "FP1 FPZ FP2 AF3 AF4 F7 F5 F3 F1 FZ F2 F4 F6 F8 FT7 FC5 FC3 FC1 FCZ FC2 FC4 FC6 FT8 T7 C5 C3 C1 CZ C2 C4 C6 T8 M1 TP7 CP5 CP3 CP1 CPZ CP2 CP4 CP6 TP8 M2 P7 P5 P3 P1 PZ P2 P4 P6 P8 PO7 PO5 PO3 POZ PO4 PO6 PO8 CB1 O1 OZ O2 CB2".split()
CHANNELS_60 = [c for c in CHANNELS_64 if c not in DROP]
EXCLUDE = {"sub-038", "sub-016", "sub-033", "sub-107", "sub-024", "sub-034", "sub-046", "sub-067", "sub-076", "sub-080", "sub-052", "sub-091"}
EC_CODES, EO_CODES = {1, 3, 5}, {2, 4, 6}
SEG_SEC = 60.0

log = logging.getLogger("preprocess")

def block_onsets(events_tsv: Path) -> dict[int, float]:
    """First onset (s) of each block code 1-6, identified by numeric event codes."""
    ev = pd.read_csv(events_tsv, sep="\t")
    values = pd.to_numeric(ev["value"], errors="coerce")
    keep = values.isin(list(range(1, 7)) + list(range(11, 17)))
    ev = ev[keep].copy()
    ev["block"] = values[keep].astype(int) % 10

    # Cross-check against text labels only where they exist
    tt = ev["trial_type"].astype(str)
    labelled = tt.str.startswith("Eyes")
    expected = np.where(ev["block"].isin(EC_CODES), "Eyes Closed", "Eyes Open")
    if not all(a.startswith(b) for a, b, m in zip(tt, expected, labelled) if m):
        raise ValueError("Event code and eyes-open/closed labels disagree")

    onsets = ev.groupby("block")["onset"].min().to_dict()
    if set(onsets) != set(range(1, 7)) or not np.isfinite(list(onsets.values())).all():
        raise ValueError("Expected six finite block onsets")
    ordered = sorted(onsets.values())
    if min(ordered) < 0 or any(b - a < SEG_SEC for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Negative or overlapping 60-second block onsets")
    return onsets



def preprocess_subject(set_path: Path, events_tsv: Path, l_freq: float, h_freq: float,
                       sfreq_out: float, channels: int = 60) -> dict:
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="error")
    desired = CHANNELS_60 if channels == 60 else CHANNELS_64
    names = [c.upper() for c in raw.ch_names]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate channel names")
    raw.rename_channels(dict(zip(raw.ch_names, names)))
    missing = set(desired) - set(raw.ch_names)
    if missing:
        raise ValueError(f"Missing required electrodes: {sorted(missing)}")
    raw.pick(desired)
    raw.reorder_channels(desired)
    if raw.info['bads']:
        raise ValueError(f"Flagged bad channels require a predefined handling policy: {raw.info['bads']}")
    raw.set_channel_types({c: "eeg" for c in raw.ch_names})
    raw.filter(l_freq, h_freq, fir_design="firwin", verbose="error")
    raw.set_eeg_reference("average", projection=False, verbose="error")
    raw.resample(sfreq_out, verbose="error")

    n_seg = int(round(SEG_SEC * sfreq_out))
    data = raw.get_data()
    segs, blocks, conds = [], [], []
    for blk, onset in sorted(block_onsets(events_tsv).items()):
        start = int(round(onset * sfreq_out))
        if start + n_seg > data.shape[1]:
            raise ValueError(f"{set_path.name} block {blk} runs past recording end")
        segs.append(data[:, start:start + n_seg])
        blocks.append(blk)
        conds.append("EC" if blk in EC_CODES else "EO")
    if not np.isfinite(segs).all():
        raise ValueError("Nonfinite EEG")
    return dict(X=np.stack(segs).astype(np.float32), block=np.array(blocks),
                cond=np.array(conds), ch_names=np.array(raw.ch_names), sfreq=sfreq_out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bids-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/derivatives"))
    ap.add_argument("--run", default="01")
    ap.add_argument("--l-freq", type=float, default=1.0)
    ap.add_argument("--h-freq", type=float, default=40.0)
    ap.add_argument("--sfreq", type=float, default=250.0)
    ap.add_argument("--subjects", nargs="*", help="e.g. sub-001 sub-002 (default: all)")
    ap.add_argument("--channels", type=int, choices=[60, 64], default=60,
                    help="60 excludes M1/M2/CB1/CB2; 64 includes them, always excludes EOG")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Use an empty preprocessing output folder")
    if not (0 < args.l_freq < args.h_freq < args.sfreq / 2):
        raise ValueError("Require 0 < low < high < output Nyquist")
    args.out.mkdir(parents=True, exist_ok=True)

    subs = args.subjects or sorted(p.name for p in args.bids_root.glob("sub-*") if p.is_dir())
    if not subs or len(subs) != len(set(subs)):
        raise ValueError("Supply a nonempty, unique subject list")
    completed = []
    for sub in subs:
        if sub in EXCLUDE:
            log.info("%s excluded (invalid participant)", sub)
            continue
        stem = args.bids_root / sub / "eeg" / f"{sub}_task-Rest_run-{args.run}"
        set_path, ev_path = Path(f"{stem}_eeg.set"), Path(f"{stem}_events.tsv")
        if not set_path.exists() or not ev_path.exists():
            raise FileNotFoundError(f"{sub}: missing {set_path} or {ev_path}")
        out = preprocess_subject(set_path, ev_path, args.l_freq, args.h_freq, args.sfreq, args.channels)
        np.savez_compressed(args.out / f"{sub}.npz", **out)
        completed.append(sub)
        log.info("%s: %d blocks %s, shape %s", sub, len(out["block"]),
                 "".join(out["cond"]), out["X"].shape)
    if not completed:
        raise ValueError("No included subjects")
    (args.out / 'preprocessing.json').write_text(json.dumps({
        'subjects': completed, 'excluded': sorted(EXCLUDE), 'channels': args.channels,
        'channel_order': CHANNELS_60 if args.channels == 60 else CHANNELS_64,
        'run': args.run, 'low_hz': args.l_freq, 'high_hz': args.h_freq,
        'sfreq': args.sfreq, 'seconds': SEG_SEC, 'reference': 'average',
        'ica': False, 'mne': mne.__version__, 'numpy': np.__version__,
        'pandas': pd.__version__}, indent=2) + '\n')


if __name__ == "__main__":
    main()
