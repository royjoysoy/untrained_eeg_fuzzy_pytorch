"""Preprocess every participant once and cache their windows (data/cache by default).

Uses the `cleaning` setting of the config: minimal (seconds per participant) or
full (bad channels, ICA + ICLabel, autoreject: a few minutes each, so use --workers).
Participants already in the cache are skipped.

    python scripts/01_build_cache.py
    python scripts/01_build_cache.py --config configs/full.yaml --workers 16
"""

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.data import load_config, load_participants, load_windows

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default=None)
parser.add_argument("--workers", type=int, default=1, help="parallel processes (default 1)")
args = parser.parse_args()
cfg = load_config(args.config) if args.config else load_config()
pids = load_participants(cfg)["participant_id"].tolist()


def build(pid):
    start = time.time()
    try:
        X, cond = load_windows(cfg, pid, build=True)
    except Exception as err:  # report and go on; a rerun retries only the missing ones
        return pid, None, f"{type(err).__name__}: {err}", time.time() - start
    return pid, (int((cond == "closed").sum()), int((cond == "open").sum())), None, time.time() - start


print(f"{len(pids)} participants, cleaning: {cfg.get('cleaning', 'minimal')}, cache: {cfg['cache_dir']}", flush=True)
pool = Pool(args.workers) if args.workers > 1 else None
failed = []
for i, (pid, counts, error, seconds) in enumerate((pool.imap_unordered if pool else map)(build, pids), 1):
    if error:
        failed.append(pid)
        print(f"[{i}/{len(pids)}] {pid} FAILED ({seconds:.0f} s): {error}", flush=True)
    else:
        print(f"[{i}/{len(pids)}] {pid}: {counts[0]} closed / {counts[1]} open windows ({seconds:.0f} s)", flush=True)
if pool:
    pool.close()
print(f"done; {len(failed)} failed{': ' + ' '.join(failed) if failed else ''}")
sys.exit(1 if failed else 0)
