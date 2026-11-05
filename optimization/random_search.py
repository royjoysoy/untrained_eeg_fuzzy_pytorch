#!/usr/bin/env python
"""Sample a small random search over the 2-3 factors with the biggest ablation effects.

Run AFTER the one-factor ablation, once the factors are chosen (by Roy):

    python optimization/random_search.py --space space.json --n 12 --out search_configs.txt
    # space.json, e.g. {"kernel1": [13, 25, 51], "nonlinearity": ["elu", "relu"],
    #                   "readout": ["mean", "meanstd"]}

Then extract the listed configs for every seed and dataset
(extract.py cnn --configs $(cat search_configs.txt)) and run
evaluate_dataset.py --nested --search-list search_configs.txt: the best search config is
chosen inside each outer training set (nested), never on the test folds; summarize.py
reports it as cnn_search_nested.
"""

import argparse
import itertools
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimization.arch import ArchConfig  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--space", type=Path, required=True)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    space = json.loads(a.space.read_text())
    for k, v in space.items():  # tuples for multiscale
        space[k] = [tuple(x) if isinstance(x, list) else x for x in v]
    grid = [dict(zip(space, vals)) for vals in itertools.product(*space.values())]
    random.Random(a.seed).shuffle(grid)
    names = []
    for kw in grid:
        try:
            cfg = replace(ArchConfig(), **kw)
        except TypeError:
            continue
        names.append(cfg.name())
        if len(names) == a.n:
            break
    if "default" not in names:
        names.append("default")  # the current config is always a candidate
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.write_text("\n".join(names) + "\n")
    print(f"{len(names)} of {len(grid)} combinations -> {a.out}")


if __name__ == "__main__":
    main()
