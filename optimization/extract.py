#!/usr/bin/env python
"""Block features for the optimization study, in the phase-1 file layout.

CNN configs (one process: one dataset, one or more configs, all their seeds):
    python optimization/extract.py cnn --dataset ds003478 --data-root $S/data \
        --out results/optimization/11-5-26 --configs grid --seeds 0-29
    python optimization/extract.py cnn ... --grid-index 5      # one grid config (SLURM arrays)
Handcrafted families:
    python optimization/extract.py handcrafted --dataset ds003478 --data-root $S/data --out ... --jobs 8
Step (b), one preprocessing level (optimization/prep.py), written under <out>/prep/<level>/:
    python optimization/extract.py cnn ... --configs default --prep win4 [--cache-root $C]
Step (a'), one readout level (optimization/readout.py), written under <out>/readout/<level>/:
    python optimization/extract.py cnn ... --configs default --readout logvar

Writes <out>/<dataset>/features/cnn/<config>/seed###.npz and
<out>/<dataset>/features/handcrafted/<family>.npz. Existing files are skipped (resumable),
never overwritten. No labels are read.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimization.common import block_features, dataset_config, embed, load_windows, save_features  # noqa: E402
from optimization.arch import ArchConfig, ConfigurableCNN1d, ablation_grid, from_name, prepare_input  # noqa: E402
from optimization.runinfo import run_info  # noqa: E402
from optimization import prep, readout  # noqa: E402


def seed_range(text):
    a, _, b = text.partition("-")
    return list(range(int(a), int(b or a) + 1))


def run_cnn(args, cfg, participants, windows, per_block, keep=None):
    pids = participants["participant_id"].tolist()
    n_chans = windows[pids[0]][0].shape[1]
    n_windows = sum((len(windows[p][0]) // per_block) * per_block for p in pids)
    if args.grid_index is not None:
        configs = [ablation_grid()[args.grid_index]]
    elif args.configs == ["grid"]:
        configs = ablation_grid()
    else:
        grid = {c.name(): (f, v, c) for f, v, c in ablation_grid()}  # keep grid labels when names match
        configs = [grid.get(n, ("manual", n, from_name(n))) for n in args.configs]
    for fac, val, arch in configs:
        inputs = None  # built once per config (unc3 depends only on the config)
        agg = "mean"
        if args.prep:  # step (b): factor/value describe the preprocessing level, the folder the arch
            fac, val = prep.LEVELS[args.prep].factor, args.prep
            run = prep.arch_for(arch, args.prep)  # sf256_k2 doubles the first kernel and the pooling
            agg = prep.LEVELS[args.prep].agg
        elif args.readout:  # step (a'): same weights, another readout
            fac, val = readout.LEVELS[args.readout].factor, args.readout
            run = readout.arch_for(arch, args.readout, cfg["sfreq"])
            agg = readout.LEVELS[args.readout].agg
        else:
            run = arch
        for seed in args.seeds:
            out = args.out / args.dataset / "features" / "cnn" / arch.name() / f"seed{seed:03d}.npz"
            if out.exists():
                continue
            if inputs is None:
                t_in = time.time()
                inputs = {p: prepare_input(windows[p][0], run) for p in pids}
                input_seconds = time.time() - t_in  # unc3 construction; ~0 for raw input
            t0 = time.time()
            model = ConfigurableCNN1d(n_chans, run, seed=seed)
            rows = [block_features(embed(model, inputs[p]), windows[p][1], per_block,
                                   None if keep is None else keep[p], agg) for p in pids]
            meta = dict(kind="cnn", config=arch.name(), factor=fac, value=val, seed=seed,
                        feature_dim=int(rows[0][0].shape[1]),
                        receptive_field_s=run.receptive_field() / cfg["sfreq"],
                        extract_seconds=time.time() - t0, input_seconds=input_seconds,
                        n_windows=n_windows, n_params=sum(q.numel() for q in model.parameters()),
                        torch=torch.__version__, threads=torch.get_num_threads())
            if args.prep:
                meta.update(prep=args.prep, arch_run=run.name(), sfreq=cfg["sfreq"],
                            window_s=cfg["window_s"], **prep.rejection_stats(keep))
            if args.readout:
                meta.update(readout=args.readout, arch_run=run.name(), agg=agg)
                spd = model.spd_blocks(meta["feature_dim"] // (2 if agg == "mean_sd" else 1))
                if spd:  # evaluate.RiemannTangent projects these columns inside the training folds
                    meta["spd_blocks"] = spd
            save_features(out, rows, pids, "untrained", meta)
            print(f"{args.dataset} {arch.name():40s} seed{seed:03d} {meta['feature_dim']:6d} feats "
                  f"{meta['extract_seconds']:.1f}s", flush=True)


def _hand_one(X, cond, per_block, sfreq, names, keep=None):
    from optimization.handcrafted import block_features as hand
    n = len(X) // per_block
    blocks = X[: n * per_block].reshape(n, per_block, X.shape[1], X.shape[2])
    K = np.ones((n, per_block), bool) if keep is None else keep[: n * per_block].reshape(n, per_block)
    out = [hand(np.concatenate(list(b[k]), axis=1), sfreq, names) for b, k in zip(blocks, K)]  # kept windows -> block
    return out, cond[::per_block][:n]


def run_handcrafted(args, cfg, participants, windows, per_block, keep=None):
    from joblib import Parallel, delayed
    from optimization.handcrafted import FAMILIES
    root = args.out / args.dataset / "features" / "handcrafted"
    if all((root / f"{f}.npz").exists() for f in FAMILIES):
        return
    pids = participants["participant_id"].tolist()
    n_ch = windows[pids[0]][0].shape[1]
    names = [str(c) for c in cfg["channels"]]
    if len(names) != n_ch:
        names = [f"ch{i}" for i in range(n_ch)]
    t0 = time.time()
    res = Parallel(n_jobs=args.jobs)(delayed(_hand_one)(*windows[p], per_block, cfg["sfreq"], names,
                                                        None if keep is None else keep[p]) for p in pids)
    secs = time.time() - t0
    for fam in FAMILIES:
        out = root / f"{fam}.npz"
        if out.exists():
            continue
        rows = [(np.stack([b[fam][0] for b in blocks]), cond) for blocks, cond in res]
        meta = dict(kind="handcrafted", config=fam, feature_dim=int(rows[0][0].shape[1]),
                    extract_seconds_all_families=secs, n_jobs=args.jobs,
                    n_windows=sum((len(windows[p][0]) // per_block) * per_block for p in pids))
        if args.prep:
            meta.update(prep=args.prep, sfreq=cfg["sfreq"], **prep.rejection_stats(keep))
        if fam.startswith("riemann"):
            n = n_ch
            k = n * (n + 1) // 2
            meta["spd_blocks"] = [[i * k, n] for i in range(rows[0][0].shape[1] // k)]
        save_features(out, rows, pids, "handcrafted", meta)
        print(args.dataset, fam, rows[0][0].shape, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=["cnn", "handcrafted"])
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--configs", nargs="+", default=["default"], help="'grid' or config names")
    p.add_argument("--grid-index", type=int, default=None)
    p.add_argument("--seeds", type=seed_range, default=list(range(30)))
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--prep", choices=list(prep.LEVELS), default=None,
                   help="step (b) preprocessing level; output goes to <out>/prep/<level>")
    p.add_argument("--cache-root", type=Path, default=None, help="where raw-based level caches live")
    p.add_argument("--readout", choices=list(readout.LEVELS), default=None,
                   help="step (a') readout level; output goes to <out>/readout/<level>")
    a = p.parse_args()
    if a.readout and (a.prep or a.kind != "cnn"):
        raise SystemExit("--readout: CNN features only, with the reference preprocessing (no --prep)")
    if a.prep and a.kind == "handcrafted" and not prep.for_handcrafted(a.prep):
        print(f"{a.prep}: handcrafted features equal those of the reference (or sf256); nothing to do")
        return
    out_root = a.out
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    with run_info(out_root, f"extract_{a.kind}", a.dataset) as rec:
        t0 = time.time()
        if a.prep:
            cfg, participants, windows, per_block, keep = prep.load(a.dataset, a.data_root, a.prep, a.cache_root)
            a.out = a.out / "prep" / a.prep
            rec["prep"] = a.prep
        else:
            cfg = dataset_config(a.dataset, a.data_root)
            if a.readout and not readout.applies(a.readout, cfg):
                print(f"{a.readout} does not apply to {a.dataset} (it equals the reference there)")
                return
            participants, windows, per_block = load_windows(cfg)
            keep = None
            if a.readout:
                a.out = a.out / "readout" / a.readout
                rec["readout"] = a.readout
        rec["load_seconds"] = round(time.time() - t0, 2)
        rec["configs"] = a.configs if a.grid_index is None else [ablation_grid()[a.grid_index][2].name()]
        rec["seeds"] = a.seeds
        (run_cnn if a.kind == "cnn" else run_handcrafted)(a, cfg, participants, windows, per_block, keep)


if __name__ == "__main__":
    main()
