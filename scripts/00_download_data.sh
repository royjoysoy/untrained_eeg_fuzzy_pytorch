#!/bin/bash
# Download OpenNeuro ds003478 v1.1.0 (resting EEG, depression; ~10 GB, no account needed).
#
#   bash scripts/00_download_data.sh /path/to/ds003478
#   export UNTRAINED_EEG_DATA=/path/to/ds003478
#
# Needs the AWS CLI (pip install awscli). On Rorqual, please download once to a
# shared /project folder instead of everyone keeping their own copy (docs/rorqual.md).
set -euo pipefail

DEST=${1:?usage: bash scripts/00_download_data.sh /path/to/ds003478}
VERSION=1.1.0

aws s3 sync --no-sign-request s3://openneuro.org/ds003478 "$DEST"

# S3 always serves the latest snapshot. Check that it is still v1.1.0; if not,
# get the exact version with DataLad instead:
#   datalad install https://github.com/OpenNeuroDatasets/ds003478.git && cd ds003478
#   git checkout 1.1.0 && datalad get .
latest=$(head -1 "$DEST/CHANGES" | cut -d' ' -f1)
[ "$latest" = "$VERSION" ] || echo "WARNING: downloaded v$latest, the project uses v$VERSION (see comment above)"
n_set=$(find "$DEST" -name '*.set' | wc -l)
n_fdt=$(find "$DEST" -name '*.fdt' | wc -l)
echo "found $n_set .set and $n_fdt .fdt files (expected 243 and 243)"
echo "now run: export UNTRAINED_EEG_DATA=$DEST"
