"""Goal 1 (beginner): run one untrained CNN on ds003478 and probe high vs low BDI.

The first call also preprocesses the EEG and caches it in data/cache/
(takes a few minutes; later scripts reuse the cache).

    python scripts/01_run_untrained.py
    python scripts/01_run_untrained.py --model eegnet --condition open --seed 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.data import load_config, load_dataset
from untrained_eeg.models import MODELS
from untrained_eeg.variability import run_once

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default=None, help="YAML config (default: configs/ds003478.yaml)")
parser.add_argument("--model", default="cnn1d", choices=MODELS)
parser.add_argument("--condition", default="closed", choices=["closed", "open"])
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

cfg = load_config(args.config) if args.config else load_config()
participants, windows = load_dataset(cfg)
print(f"{len(participants)} participants: "
      f"{(participants.label == 0).sum()} low BDI, {(participants.label == 1).sum()} high BDI")
X, cond = windows[participants.participant_id[0]]
print(f"each window: {X.shape[1]} channels x {X.shape[2]} samples; "
      f"{participants.participant_id[0]} has {len(X)} windows")

result = run_once(cfg, participants, windows, args.model, args.condition, args.seed)
print(f"\n{args.model}, eyes {args.condition}, seed {args.seed}: "
      f"AUC = {result['auc']:.3f}, balanced accuracy = {result['bacc']:.3f}")
