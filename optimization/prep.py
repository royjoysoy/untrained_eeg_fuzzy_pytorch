#!/usr/bin/env python
"""Step (b): preprocessing levels, one factor at a time away from the phase-1 cache.

The architecture and the probe stay fixed; only the input changes. Reference ("ref") =
the phase-1 "standard" cache, untouched. Every other level changes ONE thing:

| factor        | levels                  | how                                                     |
|---------------|-------------------------|---------------------------------------------------------|
| window_length | win4, win10             | join neighbouring cached windows (contiguous in a block) |
| normalization | norm_subject            | z-score per channel over all of a participant's windows |
|               | norm_window             | z-score per channel within each window                  |
| artifact      | rej_p2p150              | drop windows with a peak-to-peak > 150 uV on any channel |
|               | rej_mad5                | drop windows whose worst peak-to-peak > median + 5 MAD   |
|               |                         | (robust z, per participant)                              |
| aggregation   | agg_median, agg_trim20  | block = median / 20 % trimmed mean of its windows        |
| band_pass     | lp30                    | extra zero-phase low-pass at 30 Hz on each cached block  |
|               | bp05_45                 | 0.5-45 Hz band-pass: a NEW cache, built from raw         |
| sampling_rate | sf256                   | the existing phase-1 "fm256" cache (256 Hz)              |
|               | sf256_k2                | sf256 with first kernel x2 (+1, odd) and pooling x2, so  |
|               |                         | every layer covers the same time span as at 128 Hz       |

Rejected windows: a block that would lose ALL its windows keeps them all, so every level
has the same samples as the reference (same rows, same folds). Re-referencing is not a
factor: the cache is already average-referenced, so a common average does nothing.
No labels are used anywhere here.

    python optimization/prep.py levels --dataset ds003478       # levels that apply
    python optimization/prep.py build --dataset lemon --data-root $S/data --cache-root $C
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimization.common import dataset_config  # noqa: E402
from untrained_eeg.data import load_dataset, same_as_standard, variant_config  # noqa: E402


@dataclass(frozen=True)
class Prep:
    factor: str = "reference"
    window_s: float | None = None   # new window length (s); None = as cached
    norm: str = "none"              # none | subject | window
    reject: str = "none"            # none | p2p150 | mad5
    agg: str = "mean"               # mean | median | trim20
    lowpass: float | None = None    # extra low-pass (Hz) on the cached blocks
    variant: str | None = None      # None | fm256 (existing cache) | bp05_45 (built from raw)
    kernel_x2: bool = False         # architecture follows the sampling rate


LEVELS = {
    "ref": Prep(),
    "win4": Prep("window_length", window_s=4),
    "win10": Prep("window_length", window_s=10),
    "norm_subject": Prep("normalization", norm="subject"),
    "norm_window": Prep("normalization", norm="window"),
    "rej_p2p150": Prep("artifact", reject="p2p150"),
    "rej_mad5": Prep("artifact", reject="mad5"),
    "agg_median": Prep("aggregation", agg="median"),
    "agg_trim20": Prep("aggregation", agg="trim20"),
    "lp30": Prep("band_pass", lowpass=30.0),
    "bp05_45": Prep("band_pass", variant="bp05_45"),
    "sf256": Prep("sampling_rate", variant="fm256"),
    "sf256_k2": Prep("sampling_rate", variant="fm256", kernel_x2=True),
}
RAW_VARIANTS = {"bp05_45": {"l_freq": 0.5, "h_freq": 45.0}}  # caches rebuilt from raw
P2P_UV = 150.0   # rej_p2p150 threshold, microvolts
MAD_K = 5.0      # rej_mad5 threshold, robust SDs above the participant's median


def applies(level: str, cfg: dict) -> bool:
    """Whether a level changes anything for this dataset (otherwise it equals the reference).

    window length: the new window must be longer and divide the segment (60 s resting
    blocks: 4 and 10 s; BNCI's 4 s trial: 4 s only; Sleep-EDF's 30 s epoch: none).
    artifact / aggregation: blocks of more than 2 windows (the resting datasets).
    256 Hz: datasets cached at 128 Hz (not Sleep-EDF, 100 Hz recordings).
    """
    p = LEVELS[level]
    per_block = cfg["segment_s"] / cfg["window_s"]
    if p.window_s:
        return p.window_s > cfg["window_s"] and cfg["segment_s"] % p.window_s == 0
    if p.reject != "none" or p.agg != "mean":
        return per_block > 2
    if p.variant == "fm256":
        return cfg["sfreq"] == 128
    return True


def for_handcrafted(level: str) -> bool:
    """Handcrafted features use the whole block (windows joined back), so window length and
    aggregation do not change them, and the kernel is not theirs: those levels = ref / sf256."""
    p = LEVELS[level]
    return p.factor not in ("window_length", "aggregation") and not p.kernel_x2


def for_minirocket(level: str) -> bool:
    return not LEVELS[level].kernel_x2  # sf256_k2 = sf256 for MiniRocket


def arch_for(arch, level: str):
    """The architecture actually run at this level (only sf256_k2 changes it)."""
    if not LEVELS[level].kernel_x2:
        return arch
    return replace(arch, kernel1=2 * arch.kernel1 + 1, pool=2 * arch.pool)


def level_config(dataset: str, data_root, level: str, cache_root=None) -> dict:
    """The dataset config whose cache this level reads (phase-1 config for cache-based levels)."""
    cfg = dataset_config(dataset, data_root)
    if not applies(level, cfg):  # judged on the phase-1 settings
        raise SystemExit(f"{level} does not apply to {dataset} (it would equal the reference)")
    v = LEVELS[level].variant
    if v == "fm256" and not same_as_standard(cfg, "fm256"):
        cfg = variant_config(cfg, "fm256")
    elif v in RAW_VARIANTS:
        if cache_root is None:
            raise SystemExit(f"{level}: give --cache-root (where its new cache is built)")
        cfg = {**cfg, **RAW_VARIANTS[v], "cache_dir": Path(cache_root) / dataset / v}
    return cfg


def load(dataset: str, data_root, level: str, cache_root=None, build=False):
    """cfg, participants, {pid: (X, cond)}, windows per block, {pid: kept windows} or None."""
    cfg = level_config(dataset, data_root, level, cache_root)
    participants, windows = load_dataset(cfg, build=build, verbose=False)
    per_block = int(cfg["segment_s"] / cfg["window_s"])
    p = LEVELS[level]
    if p.window_s:
        cfg = {**cfg, "window_s": p.window_s}
    windows, per_block, keep = transform(windows, per_block, cfg, p)
    return cfg, participants, windows, per_block, keep


# ---------------------------------------------------------------------------
def transform(windows: dict, per_block: int, cfg: dict, p: Prep):
    """Apply one level to every participant's cached windows (reference: returned untouched)."""
    if replace(p, factor="reference", variant=None, kernel_x2=False) == Prep():
        return windows, per_block, None  # ref, sf256, sf256_k2, bp05_45: the cache as read
    out, keep = {}, {} if p.reject != "none" else None
    for pid, (X, cond) in windows.items():
        n = len(X) // per_block * per_block  # complete blocks only (as block_features)
        X, cond = X[:n], cond[:n]
        b = per_block
        if p.window_s:  # k neighbouring windows -> one window
            k = int(round(p.window_s / (X.shape[2] / cfg["sfreq"])))
            X = X.reshape(n // k, k, X.shape[1], X.shape[2]).transpose(0, 2, 1, 3).reshape(n // k, X.shape[1], -1)
            cond, b = cond[::k], per_block // k
        if p.lowpass:
            X = lowpass_blocks(X, b, cfg["sfreq"], p.lowpass)
        if p.norm == "subject":
            X = zscore(X, axis=(0, 2))
        elif p.norm == "window":
            X = zscore(X, axis=2)
        if p.reject != "none":
            keep[pid] = artifact_mask(X, b, p.reject)
        out[pid] = (X, cond)
    return out, b, keep


def zscore(X, axis):
    X = X.astype(np.float64)
    sd = X.std(axis=axis, keepdims=True)
    return ((X - X.mean(axis=axis, keepdims=True)) / np.where(sd > 0, sd, 1.0)).astype(np.float32)


def lowpass_blocks(X, per_block, sfreq, hz):
    """Zero-phase 4th-order Butterworth low-pass on each block (windows joined back in time)."""
    from scipy.signal import butter, sosfiltfilt
    n, ch, t = X.shape
    blocks = X.reshape(n // per_block, per_block, ch, t).transpose(0, 2, 1, 3).reshape(n // per_block, ch, -1)
    sos = butter(4, hz, btype="low", fs=sfreq, output="sos")
    y = sosfiltfilt(sos, blocks.astype(np.float64), axis=-1)
    return y.reshape(n // per_block, ch, per_block, t).transpose(0, 2, 1, 3).reshape(n, ch, t).astype(np.float32)


def artifact_mask(X, per_block, rule):
    """True = keep the window. A block that would lose every window keeps them all."""
    worst = (X.max(axis=2) - X.min(axis=2)).max(axis=1)  # worst channel's peak-to-peak
    if rule == "p2p150":
        bad = worst > P2P_UV
    elif rule == "mad5":
        med = np.median(worst)
        bad = worst > med + MAD_K * 1.4826 * np.median(np.abs(worst - med))
    else:
        raise ValueError(rule)
    keep = ~bad.reshape(-1, per_block)
    keep[~keep.any(axis=1)] = True
    return keep.ravel()


def rejection_stats(keep):
    """Fraction of windows rejected (after the keep-all fallback), for the run records."""
    if keep is None:
        return {}
    return dict(rejected_fraction=float(1 - np.concatenate(list(keep.values())).mean()))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["levels", "build"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--cache-root", type=Path, default=None)
    a = ap.parse_args()
    if a.what == "levels":  # names only: no data needed
        cfg = dataset_config(a.dataset, None)
        print(" ".join(lv for lv in LEVELS if applies(lv, cfg)))
        return
    for v in RAW_VARIANTS:  # build the raw-based caches (MNE needed; existing files are kept)
        load(a.dataset, a.data_root, v, a.cache_root, build=True)
        print(f"{a.dataset}: {v} cache ready in {a.cache_root / a.dataset / v}", flush=True)


if __name__ == "__main__":
    main()
