"""Goal 2 (intermediate): seed variability.

Same data, same probe; only the random initialization of the CNN changes.
Writes results/seed/<model>_<condition>.npz with one row per seed.

    python scripts/02_seed_variability.py
    python scripts/02_seed_variability.py --models shallow --n-seeds 10
    python scripts/02_seed_variability.py --workers 16   # one run per process

Use --workers with slow builds (e.g. Fuzzy PyTorch in round-to-nearest mode,
see environment/fuzzy_arm.sh). Results are identical to the serial run.
"""

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.data import load_config, load_dataset
from untrained_eeg.models import MODELS
from untrained_eeg.variability import run_once

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default=None)
parser.add_argument("--models", nargs="+", choices=MODELS, default=None)
parser.add_argument("--n-seeds", type=int, default=None)
parser.add_argument("--workers", type=int, default=1, help="parallel processes (default 1)")
args = parser.parse_args()

cfg = load_config(args.config) if args.config else load_config()
models = args.models or cfg["models"]
n_seeds = args.n_seeds or cfg["n_seeds"]
participants, windows = load_dataset(cfg, build=False)
out_dir = cfg["results_dir"] / "seed"
out_dir.mkdir(parents=True, exist_ok=True)


def one_run(task):
    """One (model, condition, seed) run; the data is shared with the workers by fork."""
    start = time.time()
    return task, run_once(cfg, participants, windows, *task), time.time() - start


def save(model, condition, runs):
    auc = np.array([r["auc"] for r in runs])
    np.savez(out_dir / f"{model}_{condition}.npz",
             prob=np.stack([r["prob"] for r in runs]), auc=auc,
             bacc=np.array([r["bacc"] for r in runs]), seed=np.arange(n_seeds),
             participant_id=runs[0]["participant_id"], label=runs[0]["label"])
    print(f"{model:8s} {condition:6s}: AUC {auc.mean():.3f} +/- {auc.std(ddof=1):.3f} over {n_seeds} seeds",
          flush=True)


tasks = [(m, c, s) for m in models for c in cfg["conditions"] for s in range(n_seeds)]
done = {}  # (model, condition) -> {seed: result}
pool = Pool(args.workers) if args.workers > 1 else None
results = pool.imap_unordered(one_run, tasks) if pool else map(one_run, tasks)
for i, ((model, condition, seed), r, seconds) in enumerate(results, 1):
    print(f"[{i}/{len(tasks)}] {model} {condition} seed {seed}: AUC {r['auc']:.4f} ({seconds:.0f} s)", flush=True)
    runs = done.setdefault((model, condition), {})
    runs[seed] = r
    if len(runs) == n_seeds:  # all seeds of this model and condition are in: save them in seed order
        save(model, condition, [runs[s] for s in range(n_seeds)])
if pool:
    pool.close()
