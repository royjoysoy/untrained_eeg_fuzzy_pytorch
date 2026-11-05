#!/bin/bash
# Download the pretrained checkpoints into data/pretrained/ (never committed).
#   bash scripts/00_download_pretrained.sh
# EEGNetv4: 12 motor-imagery checkpoints, MIT license, https://huggingface.co/PierreGtch/EEGNetv4
# We only fetch the weights (model-params.pkl, read with torch.load(weights_only=True)),
# not kwargs.pkl: unpickling a downloaded file can run arbitrary code.
set -euo pipefail
cd "$(dirname "$0")/.."
URL=https://huggingface.co/PierreGtch/EEGNetv4/resolve/main
for ds in AlexMI BNCI2014001 BNCI2014004 BNCI2015001 BNCI2015004 Cho2017 \
          Lee2019_MI Ofner2017 PhysionetMI Schirrmeister2017 Weibo2014 Zhou2016; do
    dest=data/pretrained/eegnetv4/$ds
    mkdir -p "$dest"
    curl -sfL "$URL/EEGNetv4_$ds/model-params.pkl" -o "$dest/model-params.pkl"
    echo "downloaded $dest"
done

# ShallowFBCSPNet: braindecode tutorial checkpoints (one subject of BNCI2014_001), safetensors.
for repo in plot_bcic_iv_2a_moabb_trial plot_bcic_iv_2a_moabb_cropped; do
    dest=data/pretrained/shallow/$repo
    mkdir -p "$dest"
    for f in params.safetensors metadata.json; do
        curl -sfL "https://huggingface.co/braindecode/$repo/resolve/main/$f" -o "$dest/$f"
    done
    echo "downloaded $dest"
done

# Foundation models (src/untrained_eeg/foundation.py): braindecode's Hugging Face repositories
# (REVE: its authors' brain-bzh repositories, REVE Responsible Use License v1.0), cached in
# data/pretrained/hf. Compute nodes have no internet: the jobs read this cache offline.
mkdir -p data/pretrained/reve  # REVE's electrode position bank
curl -sfL https://huggingface.co/brain-bzh/reve-positions/resolve/main/positions.json \
    -o data/pretrained/reve/reve_positions.json
HF_HOME="$PWD/data/pretrained/hf" python3 - <<'PY'
import sys
sys.path.insert(0, "src")
from huggingface_hub import snapshot_download
from untrained_eeg.foundation import FOUNDATION

repos = {repo for _, repo, *_ in FOUNDATION.values()} | {"braindecode/signal-jepa"}
for repo in sorted(repos):
    snapshot_download(repo)
    print("downloaded", repo)
PY
