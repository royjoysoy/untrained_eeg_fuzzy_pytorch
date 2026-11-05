"""Shared plumbing: datasets, configs, cached windows and block features, as in phase 1.

Everything here reuses Yohan's phase-1 code (src/untrained_eeg) so the optimization
results are directly comparable with results/phase1/<dataset>/metrics.csv:
  * windows from the "standard" cache (60 s blocks -> 2 s windows at 128 Hz, etc.),
  * one feature vector per block = mean of its windows' features (phase1_features.block_features),
  * labels and samples per task from untrained_eeg.tasks.task_data, on float32 features as
    stored (phase1_probe.py does not cast; casting changes the probe in the 5th decimal).

Paths: --data-root points to a folder with cache/ and raw/ (e.g. the shared
/project/rrg-glatard/shared/untrained_eeg/data). Its raw/<dataset> becomes the
dataset's data_env variable unless that variable is already set.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from untrained_eeg.data import load_config, load_dataset  # noqa: E402
from untrained_eeg.models import embed  # noqa: E402

# Datasets and their configs, in the order used everywhere.
DATASETS = {
    "ds003478": "configs/ds003478.yaml",
    "ds002778": "configs/ds002778.yaml",
    "ds003490": "configs/ds003490.yaml",
    "ds004504": "configs/ds004504.yaml",
    "lemon": "configs/lemon.yaml",
    "sleep_edf": "configs/sleep_edf.yaml",
    "bnci2014_001": "configs/bnci2014_001.yaml",
}
RAW_NAME = {"ds003478": "ds003478"}  # raw/<name> when it differs from the key


def dataset_config(name: str, data_root: Path | None = None) -> dict:
    """Phase-1 config of a dataset; cache and raw data read from data_root if given."""
    cfg_path = ROOT / DATASETS[name]
    if data_root is not None:
        with open(cfg_path) as f:
            import yaml
            env = yaml.safe_load(f).get("data_env", "UNTRAINED_EEG_DATA")
        os.environ.setdefault(env, str(Path(data_root) / "raw" / RAW_NAME.get(name, name)))
    cfg = load_config(cfg_path)
    if data_root is not None:
        rel = Path(cfg["cache_dir"]).relative_to(ROOT / "data")
        cfg["cache_dir"] = Path(data_root) / rel
    return cfg


def load_windows(cfg: dict):
    """participants table, {participant_id: (X windows, cond)}, windows per block. Never builds caches."""
    participants, windows = load_dataset(cfg, build=False, verbose=False)
    per_block = int(cfg["segment_s"] / cfg["window_s"])
    return participants, windows, per_block


def block_features(feats: np.ndarray, cond: np.ndarray, per_block: int):
    """Mean of each block's windows (same as scripts/phase1_features.block_features)."""
    n = len(feats) // per_block
    return feats[: n * per_block].reshape(n, per_block, -1).mean(axis=1), cond[::per_block][:n]


def save_features(out: Path, rows, pids, group: str, meta: dict):
    """Same file layout as phase 1 (feats, cond, participant_id, group, variant) + meta JSON."""
    if out.exists():
        raise FileExistsError(out)  # never overwrite results
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, feats=np.concatenate([f for f, _ in rows]).astype(np.float32),
             cond=np.concatenate([c for _, c in rows]),
             participant_id=np.concatenate([[p] * len(c) for p, (_, c) in zip(pids, rows)]),
             group=group, variant="standard", meta=json.dumps(meta))
    tmp.replace(out)


__all__ = ["DATASETS", "ROOT", "dataset_config", "load_windows", "block_features", "save_features", "embed"]
