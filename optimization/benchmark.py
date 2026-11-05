#!/usr/bin/env python
"""Runtime estimate on the real caches, before submitting anything long.

    python optimization/benchmark.py --data-root $S/data [--datasets ds003478 lemon] [--seeds 30]

For each dataset: counts windows/blocks, times every grid config on a sample of
windows, one handcrafted block, and one c1 probe + one nested selection per task
on random features of the right shape; extrapolates to the full study. Prints
CPU-hours per stage. Run it inside a short interactive job (salloc), 1 CPU.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimization.arch import ConfigurableCNN1d, ablation_grid, prepare_input  # noqa: E402
from optimization.common import DATASETS, dataset_config, embed, load_windows  # noqa: E402
from optimization.evaluate import c1, nested  # noqa: E402
from untrained_eeg.tasks import task_data, task_labels, task_names  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--nested-seeds", type=int, default=30)
    p.add_argument("--sample-windows", type=int, default=600)
    a = p.parse_args()
    torch.set_num_threads(1)
    grid = ablation_grid()
    rows = []
    for ds in a.datasets:
        cfg = dataset_config(ds, a.data_root)
        t0 = time.time()
        participants, windows, per_block = load_windows(cfg)
        load_s = time.time() - t0
        pids = participants["participant_id"].tolist()
        n_win = sum(len(windows[q][0]) for q in pids)
        n_blk = n_win // per_block
        X = np.concatenate([windows[q][0] for q in pids])[: a.sample_windows]
        n_ch = X.shape[1]
        cnn_s = 0.0
        for fac, val, arch in grid:
            m = ConfigurableCNN1d(n_ch, arch, seed=0)
            t0 = time.time()
            embed(m, prepare_input(X, arch))
            per_win = (time.time() - t0) / len(X)
            cnn_s += per_win * n_win * a.seeds
        from optimization.handcrafted import block_features as hand
        names = [f"ch{i}" for i in range(n_ch)]
        blk = np.concatenate(list(windows[pids[0]][0][:per_block]), axis=1)
        t0 = time.time(); hand(blk, cfg["sfreq"], names); hand_s = (time.time() - t0) * n_blk

        labels, tasks = task_labels(cfg), task_names(cfg)
        rng = np.random.default_rng(0)
        feats = rng.standard_normal((n_blk, 64))
        cond = np.concatenate([windows[q][1][::per_block][: len(windows[q][0]) // per_block] for q in pids])
        pid = np.concatenate([[q] * (len(windows[q][0]) // per_block) for q in pids])
        probe_s = nest_s = 0.0
        for task in tasks:
            Xt, y, g, _ = task_data(feats, cond, pid, task, labels)
            t0 = time.time(); c1(Xt, y, g, task, cfg); one = time.time() - t0
            probe_s += one * len(grid) * a.seeds
            t0 = time.time()
            nested({"a": (Xt, ()), "b": (Xt[:, ::-1].copy(), ())}, y, g, task, cfg)
            two = time.time() - t0
            nest_s += two / 2 * len(grid) * a.nested_seeds  # cost grows with the number of candidates
        rows.append(dict(dataset=ds, participants=len(pids), windows=n_win, blocks=n_blk, channels=n_ch,
                         load_min=load_s / 60, cnn_extract_cpu_h=cnn_s / 3600,
                         handcrafted_cpu_h=hand_s / 3600, probe_cpu_h=probe_s / 3600,
                         nested_cpu_h=nest_s / 3600))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.loc["TOTAL"] = df.sum(numeric_only=True)
    print(df.round(2).to_string())
    print(f"\n{len(grid)} grid configs x {a.seeds} seeds; nested over {a.nested_seeds} seeds; 1 CPU thread.")


if __name__ == "__main__":
    main()
