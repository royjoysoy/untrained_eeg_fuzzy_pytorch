"""Group A (#30): per-participant variability, variance decomposition and data-split variability.

Reads predictions.csv.gz files written by probe_runs.py and writes, next to each:

  participant_variability.csv  (A2) one row per participant x model x condition x source:
      mean and SD of P(high BDI) across runs, significant digits, predicted-class flip.
      Sources: "seed" (weight seeds, round-to-nearest, fixed CV split), "sr" (stochastic
      rounding, weight seed 0, fixed split), "cv" (CV splits, weight seed 0, round-to-nearest).
  variance_components.csv      (A3) one row per participant x model x condition: variance
      of P(high BDI) split into weight seed, rounding seed and residual (seed x rounding
      interaction + noise), from the crossed weight-seed x rounding-seed runs.
  summary.csv                  medians over participants of the above.

Significant digits, two estimators (both base 10, relative to the mean over runs):
  sig_digits_parker  s = -log10(sd / |mean|) (Parker 1997), 15.95 when all runs agree exactly
  sig_digits_cnh     Sohier et al. 2021 (CNH, 95% probability, 95% confidence), via the
                     `significantdigits` package of the Verificarlo project, if installed

Variance decomposition (two-way crossed design, one run per cell, method of moments):
  MS_w, MS_k, MS_res from the ANOVA table; var_seed = (MS_w - MS_res) / n_k,
  var_rounding = (MS_k - MS_res) / n_w, var_residual = MS_res (negative estimates set to 0).
  The rounding seed k has no shared meaning across weight seeds, so numerical variability
  shows up as var_rounding + var_residual; var_rounding near 0 is expected.

    python scripts/variability_stats.py --a1 results/paper/A1 --a3 results/paper/A3 --a4 results/paper/A4
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import significantdigits as sdig
except ImportError:  # optional
    sdig = None

KEY = ["model", "condition", "participant_id"]


def sig_digits_parker(mean, sd):
    with np.errstate(divide="ignore", invalid="ignore"):
        s = -np.log10(sd / np.abs(mean))
    return np.where(sd == 0, 15.95, np.clip(s, 0, 15.95))


def per_participant(pred, source):
    """Mean, SD, significant digits and flip of P(high BDI) across runs, per participant."""
    rows = []
    for (model, condition), g in pred.groupby(["model", "condition"]):
        wide = g.pivot_table(index="run", columns="participant_id", values="prob")  # runs x participants
        p = wide.to_numpy()
        mean, sd = p.mean(axis=0), p.std(axis=0, ddof=1)
        out = pd.DataFrame(dict(model=model, condition=condition, source=source, participant_id=wide.columns,
                                n_runs=len(p), prob_mean=mean, prob_sd=sd,
                                sig_digits_parker=sig_digits_parker(mean, sd),
                                flip=((p > 0.5).any(axis=0) & ~(p > 0.5).all(axis=0))))
        if sdig is not None:
            s = sdig.significant_digits(p, reference=mean, axis=0, basis=10, method=sdig.Method.CNH)
            out["sig_digits_cnh"] = np.where(sd == 0, 15.95, np.clip(s, 0, 15.95))
        label = g.drop_duplicates("participant_id").set_index("participant_id")["label"]
        out["label"] = label[wide.columns].to_numpy()
        rows.append(out)
    return pd.concat(rows)


def decompose(pred):
    """Two-way crossed variance components per participant (weight seed x rounding seed)."""
    rows = []
    for (model, condition, pid), g in pred.groupby(KEY):
        table = g.pivot_table(index="weight_seed", columns="sr_seed", values="prob")
        if table.isna().any().any() or min(table.shape) < 2:
            continue
        y = table.to_numpy()
        nw, nk = y.shape
        grand = y.mean()
        ss_w = nk * ((y.mean(axis=1) - grand) ** 2).sum()
        ss_k = nw * ((y.mean(axis=0) - grand) ** 2).sum()
        ss_res = ((y - grand) ** 2).sum() - ss_w - ss_k
        ms_w, ms_k, ms_res = ss_w / (nw - 1), ss_k / (nk - 1), ss_res / ((nw - 1) * (nk - 1))
        within = y.std(axis=1, ddof=1)  # rounding SD for each weight seed
        rows.append(dict(model=model, condition=condition, participant_id=pid, n_weight=nw, n_rounding=nk,
                         var_seed=max((ms_w - ms_res) / nk, 0.0), var_rounding=max((ms_k - ms_res) / nw, 0.0),
                         var_residual=ms_res,
                         rounding_sd_min=within.min(), rounding_sd_median=np.median(within),
                         rounding_sd_max=within.max()))
    return pd.DataFrame(rows)


def load(folder):
    pred = pd.read_csv(Path(folder) / "predictions.csv.gz")
    pred["run"] = pred["weight_seed"].astype(str) + "_" + pred["sr_seed"].astype(str) + "_" + pred["cv_seed"].astype(str)
    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a1", help="probe_runs.py output with seed and sr runs (fixed CV split)")
    parser.add_argument("--a3", help="probe_runs.py output of the crossed weight x rounding runs")
    parser.add_argument("--a4", help="probe_runs.py output of weight seed 0 RN over many CV splits")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if sdig is None:
        print("significantdigits not installed: sig_digits_cnh left out")

    parts = []
    if args.a1:
        pred = load(args.a1)
        parts.append(per_participant(pred[pred.source == "seed"], "seed"))
        parts.append(per_participant(pred[(pred.source == "sr") & (pred.weight_seed == 0)], "sr"))
    if args.a4:
        parts.append(per_participant(load(args.a4), "cv"))
    summary = []
    if parts:
        pv = pd.concat(parts)
        pv.to_csv(out / "participant_variability.csv", index=False, float_format="%.6g")
        cols = ["prob_sd", "sig_digits_parker"] + (["sig_digits_cnh"] if "sig_digits_cnh" in pv else [])
        s = pv.groupby(["model", "condition", "source"]).agg(
            n_runs=("n_runs", "first"), **{f"median_{c}": (c, "median") for c in cols},
            flip_rate=("flip", "mean")).reset_index()
        summary.append(s)
        print(s.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    if args.a3:
        pred = load(args.a3)
        vc = decompose(pred[pred.source == "sr"])
        vc.to_csv(out / "variance_components.csv", index=False, float_format="%.6g")
        total = vc[["var_seed", "var_rounding", "var_residual"]].sum(axis=1)
        vc["share_seed"] = vc["var_seed"] / total
        s = vc.groupby(["model", "condition"]).agg(
            n_participants=("participant_id", "count"),
            median_sd_seed=("var_seed", lambda v: np.median(np.sqrt(v))),
            median_sd_numerical=("var_residual", lambda v: np.median(np.sqrt(v))),
            median_share_seed=("share_seed", "median"),
            median_ratio_rounding_sd_max_min=("rounding_sd_max",
                                              lambda v: np.median(v / vc.loc[v.index, "rounding_sd_min"].replace(0, np.nan))),
        ).reset_index()
        print(s.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
        s.to_csv(out / "variance_summary.csv", index=False, float_format="%.6g")
    if summary:
        pd.concat(summary).to_csv(out / "summary.csv", index=False, float_format="%.6g")
    print(f"wrote {out}")
