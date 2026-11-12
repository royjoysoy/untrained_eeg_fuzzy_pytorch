#!/usr/bin/env python
"""Step (a'): readout levels, one factor at a time away from each architecture's own readout.

The readout turns the time course of every filter in the last block into a few numbers per
window (then blocks average their windows, as in phase 1). Data, weights, probe and folds
stay fixed; only the readout changes. Suggested by Arel: EEG-appropriate readouts.

| level        | label | what (per filter)                                              | applies to   |
|--------------|-------|----------------------------------------------------------------|--------------|
| ref          |       | the architecture's own readout (default: mean; combined:       | all          |
|              |       | mean/std/ppv/max + off-diagonal covariance of 32 filters)      |              |
| logvar       | A1    | log-variance (log band power), last block                       | all          |
| logpow_first | A2    | first conv WITHOUT nonlinearity -> square -> mean -> log        | all          |
|              |       | (ShallowConvNet-like, no ELU)                                   |              |
| ref_logvar   | A4    | ref + A1 (does log power add anything?)                         | all          |
| subwin       | B2    | log-variance in 0.5 s sub-windows -> their mean and SD          | all          |
| across_win   | B3    | per-window log-variance -> mean AND SD over the block's windows | blocks of >2 |
|              |       |                                                                 | windows      |
| hjorth       | H1    | Hjorth activity (log-variance), mobility, complexity            | all          |
| tcov32       | C1    | covariance of 32 filters WITH diagonal, shrinkage 0.1, tangent  | all          |
|              |       | space fitted inside the training folds                          |              |
| ref_tcov32   | C2    | ref, with the covariance part replaced by C1 (default: added)   | all          |
| tcov16       | H3    | C1 with 16 filters (136 features): is dimension the problem?    | all          |

Logs are taken per window, before the block average (as ShallowConvNet does per trial).
Shrinkage uses only the window itself (its own trace): no information from other samples.
Where a level does not apply (B3 on BNCI's 2-window trials and Sleep-EDF's single epoch) it
equals the reference in the selection, as in step (b).

    python optimization/readout.py levels --dataset ds003478
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SHRINK = 0.1      # approved 11-11-26
SUBWIN_S = 0.5    # B2 sub-window length (s)


@dataclass(frozen=True)
class Readout:
    factor: str
    label: str = ""
    change: dict = field(default_factory=dict)  # ArchConfig fields set at this level
    agg: str = "mean"                            # block aggregation of window features
    multi_window: bool = False                   # needs blocks of more than 2 windows


SPD = dict(cov_mode="spd", shrink=SHRINK)
LEVELS = {
    "ref": Readout("reference"),
    "logvar": Readout("spectral", "A1", dict(readout="logvar", cov_channels=0)),
    "logpow_first": Readout("spectral", "A2", dict(readout="logpow", readout_from="first_lin", cov_channels=0)),
    "ref_logvar": Readout("spectral", "A4", dict(readout_extra="logvar")),
    "subwin": Readout("temporal", "B2", dict(readout="subwin", cov_channels=0)),
    "across_win": Readout("temporal", "B3", dict(readout="logvar", cov_channels=0), agg="mean_sd",
                          multi_window=True),
    "hjorth": Readout("temporal", "H1", dict(readout="hjorth", cov_channels=0)),
    "tcov32": Readout("tangent", "C1", dict(readout="none", cov_channels=32, **SPD)),
    "ref_tcov32": Readout("tangent", "C2", dict(cov_channels=32, **SPD)),
    "tcov16": Readout("tangent", "H3", dict(readout="none", cov_channels=16, **SPD)),
}


def applies(level: str, cfg: dict) -> bool:
    """B3 needs several windows per block (60 s resting blocks: 30); everything else applies."""
    return not LEVELS[level].multi_window or cfg["segment_s"] / cfg["window_s"] > 2


def arch_for(arch, level: str, sfreq: float):
    """The architecture actually run at this level (same weights: the readout draws none)."""
    lv = LEVELS[level]
    if not lv.change:
        return arch
    if arch.readout_blocks != "last":
        raise ValueError("readout levels assume one readout block (readout_blocks='last')")
    change = dict(lv.change)
    if change.get("readout") == "subwin":  # samples per 0.5 s at the readout layer's rate
        change["subwin"] = max(2, int(round(SUBWIN_S * sfreq / arch.pool ** (arch.n_blocks - 1))))
    return replace(arch, **change)


def for_handcrafted(level: str) -> bool:  # the readout is a CNN choice: baselines stay at ref
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
