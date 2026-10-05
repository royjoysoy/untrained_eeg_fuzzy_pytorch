"""Goal 3 (intermediate): numerical variability with Fuzzy PyTorch (MCA).

The seed is fixed (mca_seed in the config). Inside the Fuzzy PyTorch container,
floating-point operations are randomly perturbed (Monte Carlo Arithmetic), so
each *process* gives one MCA sample. Run this script once per sample:

    apptainer exec <fuzzy-image.sif> python scripts/03_mca_variability.py --sample 0
    apptainer exec <fuzzy-image.sif> python scripts/03_mca_variability.py --sample 1
    ...  (slurm/mca_variability.sh does this as an array job)

Writes results/mca/<model>_<condition>/sample-<NNN>.npz.

Sanity check: outside the container every sample should be identical
(zero variability), which confirms the pipeline itself is deterministic.
The EEG cache must already exist (run 01 first, outside the container),
so that preprocessing is not perturbed.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.data import load_config, load_dataset
from untrained_eeg.models import MODELS
from untrained_eeg.variability import run_once

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--sample", type=int, required=True, help="MCA sample index (used for the file name)")
parser.add_argument("--config", default=None)
parser.add_argument("--models", nargs="+", choices=MODELS, default=None)
args = parser.parse_args()

cfg = load_config(args.config) if args.config else load_config()
participants, windows = load_dataset(cfg, build=False, verbose=False)

for model in args.models or cfg["models"]:
    for condition in cfg["conditions"]:
        r = run_once(cfg, participants, windows, model, condition, cfg["mca_seed"])
        out = cfg["results_dir"] / "mca" / f"{model}_{condition}" / f"sample-{args.sample:03d}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, sample=args.sample, seed=cfg["mca_seed"], **r)
        print(f"sample {args.sample}: {model:8s} {condition:6s} AUC {r['auc']:.6f}")
