"""Figures of the paper, from saved phase 1, phase 2 and permutation results (no model is run).

  fig1_phase1.pdf      AUC of every instance on every task (dots), one row per extractor
                       grouped by kind; the large marker is the median instance, filled when
                       it beats chance (median-instance permutation test, max-T per task;
                       first instance on Sleep-EDF, where the all-instance test was not run)
  fig2_ensembles.pdf   mean AUC of ensembles of K instances (averaged predictions) against K,
                       band = SD over draws, for the untrained ShallowFBCSPNet, scattering and
                       the EEGNetv4 checkpoints; dashed line = best single foundation model

Colours: one per kind of extractor (untrained CNN, scattering, pretrained CNN, foundation
model), the same in both figures; every extractor is also named on the axis.
Reads results/phase1/<dataset>/metrics.csv, results/phase2/<dataset>/summary.csv and
results/stats/permutation_[first_]maxt_<dataset>.csv. Writes PDF and PNG to paper/figures/.

    python scripts/paper_figures.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.data import REPO

TASKS = [  # (dataset, task, panel title), in the paper's order
    ("ds003478", "ec_eo", "Eyes closed vs open"),
    ("ds003478", "bdi_closed", "BDI, eyes closed"),
    ("ds003478", "bdi_open", "BDI, eyes open"),
    ("ds004504", "ad_vs_control", "Alzheimer's vs ctrl"),
    ("ds004504", "ftd_vs_control", "FTD vs ctrl"),
    ("ds002778", "pd_vs_control", "Parkinson's vs ctrl"),
    ("lemon", "age_closed", "Age, eyes closed"),
    ("lemon", "age_open", "Age, eyes open"),
    ("bnci2014_001", "mi_left_right", "Motor imagery L/R"),
    ("sleep_edf", "sleep_stage", "Sleep staging"),
]
KINDS = {  # group in metrics.csv: (label, colour = categorical slots 1-4 of the dataviz palette)
    "untrained": ("Untrained CNN", "#2a78d6"),
    "wavelet": ("Scattering", "#eb6834"),
    "pretrained": ("Pretrained CNN", "#1baf7a"),
    "foundation": ("Foundation model", "#eda100"),
}
EXTRACTORS = {  # extractor: axis label, top to bottom within each kind
    "cnn1d": "Small 1D CNN", "shallow": "ShallowFBCSPNet", "eegnet": "EEGNet",
    "eegnet_3ch": "EEGNet, 3 ch", "shallow_bcic2a": "Shallow, 2a input",
    "scattering": "Scattering (J 3-8)",
    "eegnetv4": "EEGNetv4 (pretr.)", "shallow_pretrained": "Shallow (pretr.)",
    "eegpt": "EEGPT", "labram": "LaBraM", "cbramod": "CBraMod",  # BENDR: left out of the analyses
    "codebrain": "CodeBrain", "eegdino_small": "EEG-DINO S", "eegdino_medium": "EEG-DINO M",
    "signal_jepa": "Signal-JEPA", "reve_base": "REVE base", "reve_large": "REVE large", "neurogpt": "Neuro-GPT",
}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def phase1(ds):
    """Instance AUCs at the fixed CV split (cv_seed 0, as in every config)."""
    m = pd.read_csv(REPO / "results" / "phase1" / ds / "metrics.csv")
    return m[m.cv_seed == 0]


def significant(ds):
    """(extractor, task) -> beats chance: median-instance test (max-T per task), else the first
    instance (Sleep-EDF, where the all-instance test was not run)."""
    stats = REPO / "results" / "stats"
    f = stats / f"permutation_maxt_{ds}.csv"
    if f.exists():
        s = pd.read_csv(f)
        s = s[s.statistic == "median"]
    else:
        s = pd.read_csv(stats / f"permutation_first_maxt_{ds}.csv")
    return {(r.extractor, r.task): bool(r.significant) for r in s.itertuples()}


def fig1(out):
    rows = list(EXTRACTORS)
    kind_of = {}
    fig, axes = plt.subplots(2, 5, figsize=(7.2, 5.2), sharey=True)
    for ax, (ds, task, title) in zip(axes.flat, TASKS):
        m, sig = phase1(ds), significant(ds)
        m = m[m.task == task]
        for y, ext in enumerate(rows):
            g = m[m.extractor == ext]
            if g.empty:
                continue
            kind_of[ext] = g.group.iloc[0]
            colour = KINDS[kind_of[ext]][1]
            ax.scatter(g.auc, [y] * len(g), s=4, color=colour, alpha=0.35, linewidths=0, zorder=2)
            filled = sig.get((ext, task), False)
            ax.scatter([g.auc.median()], [y], s=22, zorder=3, linewidths=1, edgecolors=colour,
                       facecolors=colour if filled else "white")
        for y, (a, b) in enumerate(zip(rows, rows[1:])):  # thin rule between kinds
            if a in kind_of and b in kind_of and kind_of[a] != kind_of[b]:
                ax.axhline(y + 0.5, color=GRID, lw=0.6, zorder=0)
        ax.axvline(0.5, color=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=1)
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        ax.set_title(title, color=TEXT)
        ax.set_xlim(0.25, 1.02)
        ax.set_xticks([0.3, 0.5, 0.7, 0.9])
        ax.tick_params(axis="y", length=0)
    axes[0, 0].set_yticks(range(len(rows)), [EXTRACTORS[e] for e in rows])
    axes[0, 0].invert_yaxis()
    for ax in axes[1]:
        ax.set_xlabel("AUC")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l) for l, c in KINDS.values()]
    handles += [plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=MUTED,
                           label="Median instance, not above chance"),
                plt.Line2D([], [], marker="o", ls="", color=MUTED, label="Median instance, above chance")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.6, h_pad=1.0)
    save(fig, out / "fig1_phase1")


def fig2(out):
    pools = [("shallow", "ShallowFBCSPNet (untrained)", "untrained"),
             ("scattering", "Scattering", "wavelet"),
             ("eegnetv4", "EEGNetv4 checkpoints", "pretrained")]
    fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.6), sharex=True)
    for ax, (ds, task, title) in zip(axes.flat, TASKS):
        s = pd.read_csv(REPO / "results" / "phase2" / ds / "summary.csv")
        s = s[(s.task == task) & (s.method == "average")]
        for pool, _, kind in pools:
            g = s[s.pool == pool].sort_values("k")
            if g.empty:
                continue
            c = KINDS[kind][1]
            ax.fill_between(g.k, g.auc_mean - g.auc_sd.fillna(0), g.auc_mean + g.auc_sd.fillna(0),
                            color=c, alpha=0.15, linewidth=0)
            ax.plot(g.k, g.auc_mean, color=c, lw=1.5)
        m = phase1(ds)
        fm = m[(m.task == task) & (m.group == "foundation") & m.extractor.isin(EXTRACTORS)]
        if len(fm):
            best = fm.loc[fm.auc.idxmax()]
            ax.axhline(best.auc, color=KINDS["foundation"][1], lw=1.2, ls=(0, (4, 2)))
            ax.annotate(EXTRACTORS[best.extractor], (30, best.auc), xytext=(-1, 1.5), textcoords="offset points",
                        ha="right", va="bottom", fontsize=6, color=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.margins(y=0.15)
        ax.set_title(title, color=TEXT)
        ax.set_xlim(1, 30)
        ax.set_xticks([1, 10, 20, 30])
    for ax in axes[1]:
        ax.set_xlabel("Ensemble size K")
    for ax in axes[:, 0]:
        ax.set_ylabel("AUC")
    handles = [plt.Line2D([], [], color=KINDS[k][1], lw=1.5, label=l) for _, l, k in pools]
    handles.append(plt.Line2D([], [], color=KINDS["foundation"][1], lw=1.2, ls=(0, (4, 2)),
                              label="Best single foundation model"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.8, h_pad=1.0)
    save(fig, out / "fig2_ensembles")


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", stem.with_suffix(".pdf"))


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", type=Path, default=REPO / "paper" / "figures")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
fig1(args.out)
fig2(args.out)
