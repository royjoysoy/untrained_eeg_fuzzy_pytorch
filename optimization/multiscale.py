#!/usr/bin/env python
"""Step (a''): multiscale readout and filter-bank input, one factor at a time.

Starting point (fixed since step a'): the "combined" architecture with the Hjorth readout,
BASE = n_filters=256, readout=hjorth (2 conv blocks of 256 and 512 filters, kernel 25 / 11,
ELU, average pooling 4; Hjorth activity, mobility and complexity of every filter of the LAST
block: 512 x 3 = 1,536 features). Data, probe and folds stay fixed. Ideas from the 11-12-26
meeting with Tristan: pooling discards high frequencies, so read every block (multiscale), and
give the network band-passed copies of the signal as extra input channels (filter bank).

| level  | factor       | what changes                                                  | features |
|--------|--------------|---------------------------------------------------------------|----------|
| ref    | reference    | BASE: Hjorth of the last block (= step a' "hjorth", combined) | 1,536    |
| ms2    | multiscale   | Hjorth of blocks 1 AND 2, concatenated (same weights as ref)  | 2,304    |
| last3  | depth        | 3 blocks (256, 512, 1024 filters), last block only: control   | 3,072    |
|        |              | that separates "deeper" from "reads several blocks"           |          |
| ms3    | multiscale   | 3 blocks, Hjorth of all of them                               | 5,376    |
| fb     | filter bank  | input = raw + delta/theta/alpha/beta/gamma copies of every    | 1,536    |
|        |              | electrode (arch.filter_bank), last block                      |          |
| fb_ms2 | combination  | fb + ms2                                                      | 2,304    |

Weights: PyTorch draws them block by block, so for the same seed the first two blocks of
last3 / ms3 are those of ref. The filter bank widens the first convolution (6x input
channels), so every weight differs from ref (compared over 30 seeds, not seed by seed).
Not included (approved 11-13-26): 4 blocks, whose last block keeps 4 time points of a 2 s
window (Hjorth complexity would use 2 points).
The filter-bank copies have smaller amplitudes than the raw signal and the network has no
normalization layer; kept as is in this step (noted in the results).

    python optimization/multiscale.py levels --dataset ds003478
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE = "n_filters=256,readout=hjorth"   # combined + Hjorth (step a' decision, 11-11-26)


@dataclass(frozen=True)
class Level:
    factor: str
    change: dict = field(default_factory=dict)  # ArchConfig fields set at this level
    agg: str = "mean"                            # block aggregation of window features (unchanged)


LEVELS = {
    "ref": Level("reference"),
    "ms2": Level("multiscale", dict(readout_blocks="all")),
    "last3": Level("depth", dict(n_blocks=3)),
    "ms3": Level("multiscale", dict(n_blocks=3, readout_blocks="all")),
    "fb": Level("filter_bank", dict(input_versions="fb6")),
    "fb_ms2": Level("combination", dict(input_versions="fb6", readout_blocks="all")),
}


def applies(level: str, cfg: dict) -> bool:
    """Every level applies to every dataset (a 2 s window keeps 16 time points in block 3)."""
    return True


def arch_for(arch, level: str, sfreq: float = None):
    """The architecture actually run at this level."""
    return replace(arch, **LEVELS[level].change)


def for_handcrafted(level: str) -> bool:  # CNN choices: baselines stay at step (b)'s reference
    return False


def for_minirocket(level: str) -> bool:
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["levels"])
    p.add_argument("--dataset", required=True)
    a = p.parse_args()
    from optimization.common import dataset_config
    cfg = dataset_config(a.dataset, None)
    print(" ".join(lv for lv in LEVELS if applies(lv, cfg)))


if __name__ == "__main__":
    main()
