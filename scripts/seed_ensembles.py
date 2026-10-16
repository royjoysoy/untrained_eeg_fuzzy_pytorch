"""Group B (#31, task 7): ensembles of K random CNNs, and how their variability shrinks with K.

From the round-to-nearest runs of group A (w000_rn ... w029_rn), for each K and each of R
random draws of K distinct weight seeds (fixed RNG):
  features     average the K CNNs' participant features, then one probe
  predictions  average the K per-seed probe outputs P(high BDI)
Same probe and fixed CV split as everywhere else.

Writes <out>/predictions.csv.gz (tidy: participant x model x condition x K x draw x kind) and
<out>/summary.csv (per model x condition x kind x K: AUC mean and SD over draws, median
per-participant SD of P(high BDI) over draws, flip rate).

    python scripts/seed_ensembles.py --config configs/paper.yaml --features results/paper/gpu_features \\
        --out results/paper/B7 --k 1 2 5 10 20 --draws 50
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.cli import seed_range
from untrained_eeg.data import load_config, load_participants
from untrained_eeg.models import MODELS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_runs import participant_features, probe  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default=None)
parser.add_argument("--features", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--weight-seeds", type=seed_range, default=list(range(30)))
parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 5, 10, 20])
parser.add_argument("--draws", type=int, default=50)
parser.add_argument("--models", nargs="+", choices=MODELS, default=None)
args = parser.parse_args()

cfg = load_config(args.config) if args.config else load_config()
labels = load_participants(cfg).set_index("participant_id")["label"]
rng = np.random.default_rng(0)
rows, summary = [], []
for model in args.models or cfg["models"]:
    for condition in cfg["conditions"]:
        feats = {}  # seed -> (n_participants, n_features)
        for w in args.weight_seeds:
            f = participant_features(np.load(Path(args.features) / model / f"w{w:03d}_rn.npz"), condition)
            pids = [p for p in labels.index if p in f]
            feats[w] = np.stack([f[p] for p in pids])
        y = labels[pids].to_numpy()
        single = {w: probe(X, y, cfg["cv_seed"], cfg["cv_folds"]) for w, X in feats.items()}
        for k in args.k:
            draws = [rng.choice(args.weight_seeds, size=k, replace=False) for _ in range(args.draws)]
            for kind in ("features", "predictions"):
                probs = []
                for d in draws:
                    if kind == "features":
                        p = probe(np.mean([feats[w] for w in d], axis=0), y, cfg["cv_seed"], cfg["cv_folds"])
                    else:
                        p = np.mean([single[w] for w in d], axis=0)
                    probs.append(p)
                probs = np.array(probs)  # draws x participants
                for i, (d, p) in enumerate(zip(draws, probs)):
                    rows.append(pd.DataFrame(dict(model=model, condition=condition, kind=kind, k=k, draw=i,
                                                  seeds=" ".join(map(str, sorted(d))), participant_id=pids,
                                                  label=y, prob=p)))
                aucs = np.array([roc_auc_score(y, p) for p in probs])
                pred = probs > 0.5
                summary.append(dict(model=model, condition=condition, kind=kind, k=k, draws=args.draws,
                                    auc_mean=aucs.mean(), auc_sd=aucs.std(ddof=1),
                                    median_prob_sd=np.median(probs.std(axis=0, ddof=1)),
                                    flip_rate=(pred.any(axis=0) & ~pred.all(axis=0)).mean()))
                s = summary[-1]
                print(f"{model:8s} {condition:6s} {kind:11s} K={k:2d}: AUC {s['auc_mean']:.3f} ± {s['auc_sd']:.3f}, "
                      f"median P sd {s['median_prob_sd']:.3f}, flips {s['flip_rate']:.1%}", flush=True)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
pd.concat(rows).to_csv(out / "predictions.csv.gz", index=False, float_format="%.10g")
pd.DataFrame(summary).to_csv(out / "summary.csv", index=False, float_format="%.6g")
print(f"wrote {out}")
