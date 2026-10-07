"""Group A (#30): random-CNN features of every window on the GPU (Turbulence), one file per run.

A run is one CNN (weight seed w) compiled once by IREE for the GPU, either with ordinary
round-to-nearest ("rn") or with stochastic rounding of every float add/sub/mul/div/fma
inside the kernels ("sr", Turbulence op mode, one run per rounding seed k). Window
features are averaged per participant and 1-minute block and saved to

    <out>/<model>/w<WWW>_rn.npz        weight seed w, round to nearest
    <out>/<model>/w<WWW>_sr<KKK>.npz   weight seed w, stochastic-rounding seed k

Each file holds block_feats (n_rows, n_features) float64, participant_id, block, cond and
n_windows per row (a row = one participant x block). Existing files are skipped, so a run
can be stopped and resumed, or split into chunks (e.g. one --weight-seeds range per job).
The probe is not run here: see probe_runs.py.

    python scripts/extract_gpu_features.py --weight-seeds 0-29 --rn            # A1 seeds
    python scripts/extract_gpu_features.py --weight-seeds 0 --sr-seeds 0-29    # A1 rounding
    python scripts/extract_gpu_features.py --weight-seeds 0-9 --sr-seeds 0-9   # A3 crossed

Needs an op-mode Turbulence install (TURBULENCE_IREE_BUILD, TURBULENCE_CLANG) and the EEG
cache (scripts/01_*). Uses all windows (max_windows is ignored).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.cli import seed_range
from untrained_eeg.data import load_config, load_dataset
from untrained_eeg.models import MODELS, build_model

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default=None)
parser.add_argument("--models", nargs="+", choices=MODELS, default=None)
parser.add_argument("--weight-seeds", type=seed_range, required=True, help="e.g. 0-29")
parser.add_argument("--rn", action="store_true", help="round-to-nearest run for each weight seed")
parser.add_argument("--sr-seeds", type=seed_range, default=[], help="stochastic-rounding seeds, e.g. 0-29")
parser.add_argument("--out", default=None, help="default: <results_dir>/gpu_features")
parser.add_argument("--gpu-arch", default="sm_121", help="CUDA architecture (sm_121: GB10)")
parser.add_argument("--batch-size", type=int, default=512, help="windows per GPU call (compiled in)")
args = parser.parse_args()
if not args.rn and not args.sr_seeds:
    parser.error("nothing to do: give --rn and/or --sr-seeds")

from turbulence import Configuration, instrument  # noqa: E402  (slow import, after --help)

cfg = load_config(args.config) if args.config else load_config()
out_dir = Path(args.out) if args.out else cfg["results_dir"] / "gpu_features"
participants, windows = load_dataset(cfg, build=False, verbose=False)
config = Configuration(granularity="op", rounding="stochastic_rounding", target_backend="cuda",
                       gpu_arch=args.gpu_arch)

# All windows of everyone, in one array, with the participant and block of each window.
n_win = int(cfg["segment_s"] / cfg["window_s"])  # windows per complete block
X, owner, block, cond = [], [], [], []
for i, pid in enumerate(participants["participant_id"]):
    x, c = windows[pid]
    if len(x) % n_win:
        raise SystemExit(f"{pid}: {len(x)} windows is not whole blocks of {n_win}; "
                         "this script expects the minimal cache")
    X.append(x)
    owner += [i] * len(x)
    block += list(np.arange(len(x)) // n_win)
    cond += list(c)
X, owner, block, cond = np.concatenate(X), np.array(owner), np.array(block), np.array(cond)
rows = np.unique(np.stack([owner, block]), axis=1).T  # (participant, block) pairs, sorted
row_of = {tuple(r): j for j, r in enumerate(rows)}
row_index = np.array([row_of[(o, b)] for o, b in zip(owner, block)])
row_cond = np.array([cond[np.flatnonzero(row_index == j)[0]] for j in range(len(rows))])
n_windows = np.bincount(row_index, minlength=len(rows))
print(f"{len(participants)} participants, {len(X)} windows, {len(rows)} participant-blocks", flush=True)


def features(runner):
    """Window features (n_windows, n_features), float64, in fixed-size padded batches."""
    out = []
    for i in range(0, len(X), args.batch_size):
        batch = X[i : i + args.batch_size]
        n = len(batch)
        if n < args.batch_size:  # pad the last batch; windows are computed independently
            batch = np.concatenate([batch, np.zeros((args.batch_size - n, *batch.shape[1:]), batch.dtype)])
        out.append(np.asarray(runner(batch)).reshape(args.batch_size, -1)[:n])
    return np.concatenate(out).astype(np.float64)


def save(path, f, **meta):
    """Average window features per participant-block and save."""
    sums = np.zeros((len(rows), f.shape[1]))
    np.add.at(sums, row_index, f)
    np.savez(path, block_feats=sums / n_windows[:, None], n_windows=n_windows, block=rows[:, 1],
             participant_id=participants["participant_id"].to_numpy()[rows[:, 0]].astype(str),
             cond=row_cond, **meta)


for name in args.models or cfg["models"]:
    model_dir = out_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)
    for w in args.weight_seeds:
        todo_rn = args.rn and not (model_dir / f"w{w:03d}_rn.npz").exists()
        todo_sr = [k for k in args.sr_seeds if not (model_dir / f"w{w:03d}_sr{k:03d}.npz").exists()]
        if not todo_rn and not todo_sr:
            continue
        model = build_model(name, X.shape[1], X.shape[2], w)
        example = torch.zeros(args.batch_size, X.shape[1], X.shape[2])
        t = time.time()
        if todo_rn:
            runner = instrument(model, example, config=config, instrument=False).create_runner()
            save(model_dir / f"w{w:03d}_rn.npz", features(runner), weight_seed=w, sr_seed=-1, mode="rn")
        if todo_sr:
            runner = instrument(model, example, config=config, instrument=True).create_runner()
            first = None
            for k in todo_sr:
                runner.set_seed(k)
                f = features(runner)
                if first is None:  # same rounding seed twice must be bit-identical
                    runner.set_seed(k)
                    assert np.array_equal(f, features(runner)), f"{name} w{w} sr{k}: not reproducible"
                    first = k
                save(model_dir / f"w{w:03d}_sr{k:03d}.npz", f, weight_seed=w, sr_seed=k, mode="sr")
        print(f"{name} w{w}: rn={'yes' if todo_rn else 'skip'} sr={len(todo_sr)} runs "
              f"({time.time() - t:.0f} s)", flush=True)
