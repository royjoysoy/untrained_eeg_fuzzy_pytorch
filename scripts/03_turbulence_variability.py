"""Goal 3 on a GPU: numerical variability with Turbulence instead of Fuzzy PyTorch.

Same experiment as 03_mca_variability.py (fixed seed mca_seed, many runs that
differ only by floating-point rounding), but the CNN runs on an NVIDIA GPU,
compiled by IREE with Turbulence (https://github.com/yohanchatelain/turbulence).
Every float add/sub/mul/div/fma of the CNN is stochastically rounded (op mode,
"rr": the same rounding as PRISM SR in the Fuzzy PyTorch image). The probe
(scikit-learn) is not perturbed, as in the container runs.

Unlike the container, one process does all samples: each model is compiled once
and every sample only changes the rounding seed. Needs the EEG cache (run 01
first) and an op-mode Turbulence install (TURBULENCE_IREE_BUILD) that includes
yohanchatelain/turbulence#8 (shallow and eegnet do not compile for cuda without it).

    python scripts/03_turbulence_variability.py                    # n_mca_samples samples
    python scripts/03_turbulence_variability.py --samples 5 --models cnn1d

Writes results/turbulence/<model>_<condition>/sample-<NNN>.npz (same format as
results/mca/), plus reference.npz: the same GPU program without stochastic
rounding (round-to-nearest). Summarize with the seed and Fuzzy runs:
    python scripts/summarize_metrics.py
For many weight seeds x rounding seeds, see extract_gpu_features.py (PR #33).

Sanity check: sample 0 is run twice and must be bit-identical (same seed,
same rounding), which confirms the pipeline itself is deterministic.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.data import load_config, load_dataset
from untrained_eeg.models import MODELS, build_model
from untrained_eeg.variability import probe, usable

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--samples", type=int, default=None, help="number of SR samples (default: n_mca_samples)")
parser.add_argument("--models", nargs="+", choices=MODELS, default=None)
parser.add_argument("--config", default=None)
parser.add_argument("--gpu-arch", default="sm_121", help="CUDA architecture (sm_121: GB10, sm_86: RTX 30xx, ...)")
parser.add_argument("--rounding", default="stochastic_rounding", choices=["stochastic_rounding", "up_down"])
parser.add_argument("--batch-size", type=int, default=512, help="windows per GPU call (fixed, compiled in)")
args = parser.parse_args()

from turbulence import Configuration, instrument  # noqa: E402  (slow import, after --help)

cfg = load_config(args.config) if args.config else load_config()
n_samples = args.samples or cfg["n_mca_samples"]
participants, windows = load_dataset(cfg, build=False, verbose=False)
config = Configuration(granularity="op", rounding=args.rounding, target_backend="cuda", gpu_arch=args.gpu_arch)


def compile_cnn(model, n_chans, n_times, rounded):
    """The CNN as a GPU program taking (batch_size, n_chans, n_times) float32 windows."""
    example = torch.zeros(args.batch_size, n_chans, n_times)
    return instrument(model, example, config=config, instrument=rounded).create_runner()


def features(runner, X):
    """Features of every window, (n_windows, n_features) float64, in fixed-size batches."""
    out = []
    for i in range(0, len(X), args.batch_size):
        batch = X[i : i + args.batch_size]
        n = len(batch)
        if n < args.batch_size:  # pad the last batch; windows are computed independently
            batch = np.concatenate([batch, np.zeros((args.batch_size - n, *batch.shape[1:]), batch.dtype)])
        out.append(np.asarray(runner(batch)).reshape(args.batch_size, -1)[:n])
    return np.concatenate(out).astype(np.float64)


def run(runner, X, owner, people, seed=None):
    """One run: features of all windows -> mean per participant -> probe."""
    if seed is not None:
        runner.set_seed(seed)
    f = features(runner, X)
    feats = np.stack([f[owner == k].mean(axis=0) for k in range(len(people))])
    prob, auc, bacc = probe(feats, people["label"].to_numpy(), cfg)
    return dict(prob=prob, auc=auc, bacc=bacc, participant_id=people["participant_id"].to_numpy(),
                label=people["label"].to_numpy())


for name in args.models or cfg["models"]:
    for condition in cfg["conditions"]:
        people = usable(participants, windows, condition)
        # All windows of this condition in one array; owner[i] = participant of window i.
        parts = [windows[p][0][windows[p][1] == condition] for p in people["participant_id"]]
        X = np.concatenate(parts)
        owner = np.repeat(np.arange(len(parts)), [len(p) for p in parts])
        _, n_chans, n_times = X.shape
        model = build_model(name, n_chans, n_times, cfg["mca_seed"])
        out_dir = cfg["results_dir"] / "turbulence" / f"{name}_{condition}"
        out_dir.mkdir(parents=True, exist_ok=True)

        t = time.time()
        ref = run(compile_cnn(model, n_chans, n_times, rounded=False), X, owner, people)
        np.savez(out_dir / "reference.npz", seed=cfg["mca_seed"], **ref)
        runner = compile_cnn(model, n_chans, n_times, rounded=True)
        print(f"{name} {condition}: {len(X)} windows, compiled in {time.time() - t:.0f} s, "
              f"reference AUC {ref['auc']:.6f}")

        t = time.time()
        for sample in range(n_samples):
            r = run(runner, X, owner, people, seed=sample)
            if sample == 0:  # same seed again: must be bit-identical
                again = run(runner, X, owner, people, seed=0)
                assert np.array_equal(r["prob"], again["prob"]), "same seed, different results"
            np.savez(out_dir / f"sample-{sample:03d}.npz", sample=sample, seed=cfg["mca_seed"],
                     rounding=args.rounding, gpu_arch=args.gpu_arch, **r)
            print(f"sample {sample}: {name:8s} {condition:6s} AUC {r['auc']:.6f}")
        print(f"{name} {condition}: {n_samples} samples in {time.time() - t:.0f} s")
