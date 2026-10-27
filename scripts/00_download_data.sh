#!/bin/bash
# Download an OpenNeuro dataset of the project (no account needed).
#
#   bash scripts/00_download_data.sh ds004504              # into data/raw/ds004504
#   bash scripts/00_download_data.sh ds003478 /shared/dir  # a shared copy elsewhere, then
#   export UNTRAINED_EEG_DATA=/shared/dir                   # point the scripts to it
#   (one variable per dataset: data_env in its config, e.g. UNTRAINED_EEG_DS004504)
#
# Datasets: ds003478 (run-01, 8.2 GB), ds004504 (raw, 2.8 GB), ds002778 (0.57 GB); the
# files to fetch are in configs/<dataset>.yaml. The scripts also download data/raw/<dataset>
# themselves when it is missing. On Rorqual, keep one shared copy in /project (docs/rorqual.md).
# Interrupted? Run it again: files already downloaded are skipped.
set -euo pipefail
DATASET=${1:?usage: bash scripts/00_download_data.sh <ds003478|ds004504|ds002778> [destination]}
cd "$(dirname "$0")/.."
python3 - "$DATASET" "${2:-}" <<'EOF'
import sys
sys.path.insert(0, "src")
from untrained_eeg.data import REPO, _data_dir, load_config
from untrained_eeg.download import download_openneuro

dataset, dest = sys.argv[1], sys.argv[2]
cfg = load_config(REPO / "configs" / f"{dataset}.yaml")
if dest:  # a copy elsewhere, e.g. shared
    src = cfg["openneuro"]
    download_openneuro(src["id"], dest, src["include"], src["version"])
else:  # data/raw/<dataset>, with the marker that tells the scripts it is complete
    cfg["data_dir"] = None
    print(_data_dir(cfg))
EOF
