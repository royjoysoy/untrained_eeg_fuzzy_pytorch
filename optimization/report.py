#!/usr/bin/env python
"""One figure per ablation factor: per-seed AUC on every task, with reference lines.

    python optimization/report.py --out results/optimization/11-5-26

Reads <out>/summary/ (summarize.py) and the per-seed rows. Each panel is one task;
dots are weight seeds, the black bar is their mean. Reference lines (same probe, folds):
phase-1 ShallowFBCSPNet and scattering (mean over instances), the best phase-1
foundation model, the best single handcrafted family (descriptive) and MiniRocket.
Writes <out>/summary/figures/<factor>.png.
"""

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CNN = "#2a78d6"
REFS = [("phase1_shallow", "untrained ShallowFBCSPNet (phase 1)", "#eb6834", "-"),
        ("phase1_scattering", "wavelet scattering (phase 1)", INK2, "--"),
        ("phase1_best_foundation", "best foundation model (phase 1)", "#6250d6", "-."),
        ("handcrafted_best_descriptive", "best handcrafted family", "#1baf7a", ":"),
        ("minirocket", "MiniRocket (mean over seeds)", "#eda100", (0, (5, 2, 1, 2)))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dest", default="summary")
    a = p.parse_args()
    ps = pd.concat([pd.read_csv(f) for f in sorted(a.out.glob("*/per_seed.csv"))], ignore_index=True)  # grid factors only
    head = pd.read_csv(a.out / a.dest / "headline.csv").set_index(["dataset", "task"])
    cnn = ps[ps.kind == "cnn"]
    figdir = a.out / a.dest / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    tasks = list(head.index)
    for factor in sorted(set(cnn.factor) - {"default", "manual"}):
        fac = cnn[cnn.factor == factor]
        order = ["current"] + list(dict.fromkeys(fac.value))
        sub = pd.concat([cnn[cnn.config == "default"].assign(value="current"), fac])
        ncol = min(4, len(tasks))
        nrow = math.ceil(len(tasks) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * (1.2 + 0.55 * len(order)), nrow * 2.6),
                                 squeeze=False)
        handles = {}
        for ax, (ds, task) in zip(axes.flat, tasks):
            d = sub[(sub.dataset == ds) & (sub.task == task)]
            for i, v in enumerate(order):
                vals = d[d.value == v].auc.to_numpy()
                if not len(vals):
                    continue
                jit = np.linspace(-0.15, 0.15, len(vals)) if len(vals) > 1 else np.zeros(1)
                ax.scatter(i + jit, vals, s=10, color=CNN, alpha=0.6, edgecolor="none", zorder=3)
                ax.hlines(vals.mean(), i - 0.28, i + 0.28, color=INK, lw=1.8, zorder=4)
            for col, label, color, ls in REFS:
                val = head.loc[(ds, task)].get(col, np.nan)
                if pd.notna(val):
                    handles[label] = ax.axhline(val, color=color, lw=1.2, ls=ls, zorder=2)
            ax.axhline(0.5, color=GRID, lw=0.8, zorder=0)
            ax.set_title(f"{ds} · {task}", fontsize=8, color=INK)
            ax.set_xticks(range(len(order)), order, rotation=35, ha="right", fontsize=7)
            ax.tick_params(axis="y", labelsize=7, colors=INK2)
            ax.tick_params(axis="x", colors=INK2)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.grid(axis="y", color=GRID, lw=0.5)
        for ax in list(axes.flat)[len(tasks):]:
            ax.axis("off")
        for ax in axes[:, 0]:
            ax.set_ylabel("ROC-AUC", fontsize=8, color=INK2)
        n = cnn[cnn.config == "default"].seed.nunique()
        fig.suptitle(f"{factor}: dots = weight seeds (n = {n}), bar = mean", fontsize=10, color=INK)
        fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=3, frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0.05, 1, 0.97))
        fig.savefig(figdir / f"{factor}.png", dpi=200)
        plt.close(fig)
        print("wrote", figdir / f"{factor}.png")


if __name__ == "__main__":
    main()
