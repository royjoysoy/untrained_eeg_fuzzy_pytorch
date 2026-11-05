"""Does EEGPT's Parkinson's result (ds002778) depend on its assumed input unit or on medication?

EEGPT's input unit is not documented; we feed millivolts (microvolts / 1000). This recomputes
EEGPT's segment features with other divisors and probes each with the same probe and folds:
AUC on 30 CV splits and a refit permutation p (split 0, 1,000 permutations).
--session on uses the patients' on-medication recordings instead of off (controls unchanged).

    python scripts/check_eegpt_scale.py --divisors 1 100 1000 10000 --session off
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.data import load_config, load_dataset, variant_config
from untrained_eeg.foundation import HF_HOME, Foundation, variant_of
from untrained_eeg.models import embed
from untrained_eeg.tasks import probe, task_data, task_labels

TASK = "pd_vs_control"
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--divisors", type=float, nargs="+", default=[1, 100, 1000, 10000],
                    help="input = microvolts / divisor (1000 = millivolts, our default)")
parser.add_argument("--session", choices=["off", "on"], default="off")
parser.add_argument("--permutations", type=int, default=1000)
args = parser.parse_args()
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
os.environ.setdefault("HF_HOME", str(HF_HOME))

cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "ds002778.yaml")
if args.session == "on":  # patients' other session, in its own cache
    cfg = {**cfg, "sessions": {**cfg["sessions"], "pd": "on"}, "cache_dir": Path(f"{cfg['cache_dir']}_on")}
vcfg = variant_config(cfg, variant_of("eegpt"))
participants, windows = load_dataset(vcfg)
labels = task_labels(cfg)
pids = participants["participant_id"].tolist()
_, n_chans, n_times = windows[pids[0]][0].shape
model = Foundation("eegpt", cfg["channels"], n_times).eval()  # divides by 1000 itself
per_seg = int(vcfg["segment_s"] / vcfg["window_s"])

print(f"session {args.session}: {len(pids)} participants, {n_chans} channels, {n_times} samples per window")
for div in args.divisors:
    feats, cond, pid = [], [], []
    for p in pids:
        X, c = windows[p]
        f = embed(model, X * (1000 / div))
        n = len(X) // per_seg
        feats.append(f[: n * per_seg].reshape(n, per_seg, -1).mean(axis=1))
        cond += list(c[::per_seg][:n])
        pid += [p] * n
    Xp, y, groups, _ = task_data(np.concatenate(feats), np.array(cond), np.array(pid), TASK, labels)
    aucs = [probe(Xp, y, groups, TASK, s, cfg["cv_folds"])[1] for s in range(30)]
    rng = np.random.default_rng(0)
    null = np.array([probe(Xp, rng.permutation(y), groups, TASK, 0, cfg["cv_folds"])[1]
                     for _ in range(args.permutations)])
    p = (1 + (null >= aucs[0]).sum()) / (1 + args.permutations)
    print(f"  input = uV / {div:<7g} feature SD {Xp.std():.3g}  AUC {aucs[0]:.3f}  30 splits "
          f"{np.mean(aucs):.3f} ± {np.std(aucs):.3f}  null 95% {np.quantile(null, 0.95):.3f}  p {p:.4f}", flush=True)
