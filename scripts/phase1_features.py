"""Phase 1, step 1: block features of every model instance on one dataset (ds003478 by default).

An instance is one untrained model (one weight seed) or one pretrained checkpoint. Each
instance embeds every 2 s window; the windows of each 1-minute block are averaged, giving
one feature vector per block. Probing (phase1_probe.py) and ensembles (phase 2) read these.

Extractors (each with the input its weights expect):
  cnn1d, shallow, eegnet   untrained, 60 channels, 1-40 Hz, 128 Hz ("standard" input)
  eegnet_3ch               untrained EEGNet on C3, Cz, C4: the same input as eegnetv4
  eegnetv4                 pretrained EEGNet, 10 motor-imagery checkpoints (C3, Cz, C4)
  shallow_bcic2a           untrained ShallowFBCSPNet on the same input as shallow_pretrained
  shallow_pretrained       pretrained ShallowFBCSPNet, 2 checkpoints (22 channels, 250 Hz,
                           4-38 Hz, exponential moving standardization: "bcic2a" input)
  scattering               Mallat's wavelet scattering (fixed wavelets, no weights), one
                           instance per largest scale J = 3..8 ("standard" input)
  eegpt, labram, bendr, cbramod, codebrain, eegdino_small, eegdino_medium, signal_jepa,
  reve_base, reve_large,
  neurogpt                 pretrained foundation models (src/untrained_eeg/foundation.py), one
                           instance each, each with its own sampling rate, channels and window

Extractors whose input the dataset lacks (channels, or a preprocessing variant) are skipped.
Writes results/phase1/<dataset>/features/<extractor>/<instance>.npz; existing files are
skipped, so an interrupted job can be resubmitted. Data and weights are downloaded if missing
(weights: bash scripts/00_download_pretrained.sh).

    python scripts/phase1_features.py                    # everything, 30 seeds, ds003478
    python scripts/phase1_features.py --config configs/ds004504.yaml
    python scripts/phase1_features.py --extractors eegnetv4 --seeds 0-4
    python scripts/phase1_features.py --extractors reve_base reve_large --device cuda
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.data import COMMON_VARIANTS, load_config, load_dataset, same_as_standard, variant_config
from untrained_eeg.foundation import FOUNDATION, HF_HOME, Foundation, variant_of
from untrained_eeg.models import WaveletScattering, build_model, embed
from untrained_eeg.pretrained import EEGNETV4_CHANNELS, EEGNETV4_DATASETS, SHALLOW_CHECKPOINTS, load_eegnetv4, load_shallow

# Cho2017 and Ofner2017 were trained on another input scale (docs/pretrained_models.md).
EEGNETV4_USED = [d for d in EEGNETV4_DATASETS if d not in ("Cho2017", "Ofner2017")]

EXTRACTORS = {  # name: (group, preprocessing variant, channel subset or None)
    "cnn1d": ("untrained", "standard", None),
    "shallow": ("untrained", "standard", None),
    "eegnet": ("untrained", "standard", None),
    "eegnet_3ch": ("untrained", "standard", EEGNETV4_CHANNELS),
    "eegnetv4": ("pretrained", "standard", EEGNETV4_CHANNELS),
    "shallow_bcic2a": ("untrained", "bcic2a", None),
    "shallow_pretrained": ("pretrained", "bcic2a", None),
    "scattering": ("wavelet", "standard", None),
    **{key: ("foundation", variant_of(key), None) for key in FOUNDATION},
}
SCATTERING_J = range(3, 9)  # largest wavelet scale 2**J samples: 62 ms to 2 s at 128 Hz


def instances(name, seeds, cfg):
    """(instance id, function n_chans, n_times -> model) for every instance of an extractor."""
    if name == "eegnetv4":  # minus checkpoints trained on this dataset (config `exclude`)
        used = [d for d in EEGNETV4_USED if d not in cfg.get("exclude", {}).get("eegnetv4", [])]
        return [(d, lambda c, t, d=d: load_eegnetv4(d, t)) for d in used]
    if name == "shallow_pretrained":
        return [(k, lambda c, t, k=k: load_shallow(k, t)) for k in SHALLOW_CHECKPOINTS]
    if name == "scattering":
        return [(f"J{j}", lambda c, t, j=j: WaveletScattering(t, j)) for j in SCATTERING_J]
    if name in FOUNDATION:  # one pretrained model; it picks its own channels by name
        return [("pretrained", lambda c, t: Foundation(name, cfg["channels"], t))]
    arch = {"eegnet_3ch": "eegnet", "shallow_bcic2a": "shallow"}.get(name, name)
    return [(f"seed{s:03d}", lambda c, t, s=s: build_model(arch, c, t, s)) for s in seeds]


def block_features(model, windows, per_block):
    """Mean feature vector of each block: (n_blocks, n_features), plus block conditions."""
    X, cond = windows
    if isinstance(model, Foundation):  # REVE, Neuro-GPT: one participant's windows = its recording
        X = model.prepare(X)
    f = embed(model, X)
    n = len(X) // per_block
    return f[: n * per_block].reshape(n, per_block, -1).mean(axis=1), cond[::per_block][:n]


def seed_range(text):
    a, _, b = text.partition("-")
    return list(range(int(a), int(b or a) + 1))


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default=None)
parser.add_argument("--extractors", nargs="+", choices=list(EXTRACTORS), default=list(EXTRACTORS))
parser.add_argument("--seeds", type=seed_range, default=list(range(30)), help="untrained seeds, e.g. 0-29")
parser.add_argument("--device", default="cpu", help="cuda for the large foundation models (REVE large)")
parser.add_argument("--out", type=Path, default=None, help="features folder (default results/phase1/<dataset>/features)")
args = parser.parse_args()

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
os.environ.setdefault("HF_HOME", str(HF_HOME))  # foundation-model weights, read offline
cfg = load_config(args.config) if args.config else load_config()
out_root = args.out or Path(cfg["results_dir"]) / "phase1" / cfg["openneuro"]["id"] / "features"
data = {}  # the current preprocessing variant -> (participants, {participant_id: (X, cond)})
# each extractor's variant, "standard" when that is the same input (e.g. fm128 on a 128 Hz dataset)
variants = {n: "standard" if same_as_standard(cfg, v) else v for n, (_, v, _) in EXTRACTORS.items()}

for name in sorted(args.extractors, key=variants.get):  # one variant at a time
    group, _, subset = EXTRACTORS[name]
    variant = variants[name]
    if name in cfg.get("exclude", {}).get("extractors", []):
        print(f"{name}: skipped, {cfg['exclude']['reason']}", flush=True)
        continue
    have = {c.upper() for c in cfg["channels"]}
    known = variant == "standard" or variant in cfg.get("preprocessing", {}) or variant in COMMON_VARIANTS
    vcfg = variant_config(cfg, variant) if known else None
    if not known or (subset and not {c.upper() for c in subset} <= have) or vcfg["window_s"] > vcfg["segment_s"]:
        print(f"{name}: skipped, this dataset lacks its input", flush=True)
        continue
    per_block = int(vcfg["segment_s"] / vcfg["window_s"])  # windows per block (e.g. 30 of 2 s)
    if variant not in data:
        data.clear()  # keep one variant in memory (LEMON at 250 Hz is about 10 GB)
        data[variant] = load_dataset(vcfg)
    participants, windows = data[variant]
    if subset:  # channel indices in the cached order (config names are upper case)
        names = [c.upper() for c in cfg["channels"]]
        idx = [names.index(c.upper()) for c in subset]
        windows = {p: (X[:, idx], cond) for p, (X, cond) in windows.items()}
    pids = participants["participant_id"].tolist()
    _, n_chans, n_times = windows[pids[0]][0].shape
    for inst, make in instances(name, args.seeds, cfg):
        out = out_root / name / f"{inst}.npz"
        if out.exists():
            continue
        try:
            model = make(n_chans, n_times).to(args.device)
        except ValueError as e:  # e.g. a foundation model's channels missing from the dataset
            print(f"{name}: skipped, {e}", flush=True)
            continue
        rows = [block_features(model, windows[p], per_block) for p in pids]
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, feats=np.concatenate([f for f, _ in rows]).astype(np.float32),
                 cond=np.concatenate([c for _, c in rows]),
                 participant_id=np.concatenate([[p] * len(c) for p, (_, c) in zip(pids, rows)]),
                 group=group, variant=variant)
        print(f"{name:18s} {inst:18s} {rows[0][0].shape[1]:5d} features", flush=True)
