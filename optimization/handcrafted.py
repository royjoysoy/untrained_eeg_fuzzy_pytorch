"""Handcrafted EEG feature families (Step 1 baselines), computed per 60 s block.

Input: one block (n_channels, n_times) = its phase-1 windows joined back in time
(windows are contiguous and non-overlapping within a block), in the units of the
standard cache (microvolts). One vector per block, exactly where the CNN gives one;
untrained_eeg.tasks.task_data then averages blocks per participant where the task does.

Riemannian features are special: the tangent-space reference point is learned
from data, so this module only stores the per-block covariance matrices (upper
triangle). The tangent-space projection is fitted INSIDE each training fold by
`RiemannTangent` (see evaluate.py), never on all subjects.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import butter, sosfiltfilt, welch

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40)}
FMIN, FMAX = 1.0, 40.0
FAMILIES = ("bandpower_abs", "bandpower_rel", "paf_tbr", "aperiodic", "hjorth",
            "entropy", "riemann_broad", "riemann_bands")


def psd(x, sfreq):
    """Welch PSD, 2 s Hann windows (shorter if the block is), 50 % overlap -> 0.5 Hz resolution."""
    n = min(int(2 * sfreq), x.shape[1])
    f, p = welch(x, fs=sfreq, nperseg=n, noverlap=n // 2, axis=-1)
    return f, np.maximum(p, 1e-30)


def _band(f, p, lo, hi):
    m = (f >= lo) & (f < hi)
    return trapezoid(p[:, m], f[m], axis=1)


def bandpower(f, p):
    tot = _band(f, p, FMIN, FMAX)
    ab = np.stack([_band(f, p, lo, hi) for lo, hi in BANDS.values()], 1)   # (ch, 5)
    return np.log10(ab), ab / tot[:, None]


def paf_tbr(f, p, ab):
    m = (f >= 7) & (f <= 13)
    paf = f[m][np.argmax(p[:, m], axis=1)]                 # peak alpha frequency per channel
    tbr = np.log10(ab[:, 1] / ab[:, 3])                    # log theta/beta ratio per channel
    return np.concatenate([paf, tbr])


def aperiodic(f, p):
    """specparam/FOOOF 'fixed' aperiodic fit (offset, exponent) per channel, 1-40 Hz."""
    try:
        from specparam import SpectralGroupModel as Group
    except ImportError:  # older name
        from fooof import FOOOFGroup as Group
    g = Group(peak_width_limits=(1, 8), max_n_peaks=6, aperiodic_mode="fixed", verbose=False)
    g.fit(f, p, [FMIN, min(FMAX, f[-1])], n_jobs=1)
    ap = g.get_params("aperiodic_params")                  # (ch, 2): offset, exponent
    return np.concatenate([ap[:, 0], ap[:, 1]])


def hjorth(x):
    d1 = np.diff(x, axis=1)
    d2 = np.diff(d1, axis=1)
    v0, v1, v2 = x.var(1), d1.var(1), d2.var(1)
    mob = np.sqrt(v1 / v0)
    comp = np.sqrt(v2 / v1) / mob
    return np.concatenate([np.log10(v0), mob, comp])


def spectral_entropy(f, p):
    m = (f >= FMIN) & (f <= FMAX)
    q = p[:, m] / p[:, m].sum(1, keepdims=True)
    return -(q * np.log2(q)).sum(1) / np.log2(m.sum())


def permutation_entropy(x, order=3, delay=1):
    """Normalised permutation entropy per channel (Bandt & Pompe 2002)."""
    n = x.shape[1] - (order - 1) * delay
    emb = np.stack([x[:, i * delay: i * delay + n] for i in range(order)], axis=2)  # (ch, n, order)
    ranks = np.argsort(emb, axis=2)
    codes = (ranks * (order ** np.arange(order))).sum(2)                            # unique per pattern
    out = np.empty(x.shape[0])
    for c in range(x.shape[0]):
        _, cnt = np.unique(codes[c], return_counts=True)
        q = cnt / cnt.sum()
        out[c] = -(q * np.log2(q)).sum()
    from math import factorial, log2
    return out / log2(factorial(order))


def covariance_triu(x):
    """OAS-shrunk covariance (full rank despite the average reference), upper triangle."""
    from sklearn.covariance import oas
    c, _ = oas(x.T.astype(np.float64), assume_centered=False)
    return c[np.triu_indices(c.shape[0])]


def block_features(x: np.ndarray, sfreq: float, ch_names: list[str]) -> dict[str, tuple]:
    """All families for one block -> {family: (vector, names)}."""
    x = x.astype(np.float64)
    f, p = psd(x, sfreq)
    lab, rel = bandpower(f, p)
    bn = list(BANDS)
    nm = lambda pre, keys: [f"{pre}_{c}_{k}" for k in keys for c in ch_names]
    out = {
        "bandpower_abs": (lab.T.ravel(), nm("logabs", bn)),
        "bandpower_rel": (rel.T.ravel(), nm("rel", bn)),
        "paf_tbr": (paf_tbr(f, p, 10 ** lab), nm("x", ["paf", "logtbr"])),
        "aperiodic": (aperiodic(f, p), nm("ap", ["offset", "exponent"])),
        "hjorth": (hjorth(x), nm("hj", ["logactivity", "mobility", "complexity"])),
        "entropy": (np.concatenate([spectral_entropy(f, p), permutation_entropy(x)]),
                    nm("ent", ["spectral", "perm3"])),
    }
    covs = [covariance_triu(x)]
    for lo, hi in BANDS.values():
        sos = butter(4, [lo, min(hi, 0.45 * sfreq)], btype="bandpass", fs=sfreq, output="sos")
        covs.append(covariance_triu(sosfiltfilt(sos, x, axis=1)))
    n_tri = covs[0].size
    out["riemann_broad"] = (covs[0], [f"cov_broad_{i}" for i in range(n_tri)])
    out["riemann_bands"] = (np.concatenate(covs[1:]),
                            [f"cov_{b}_{i}" for b in bn for i in range(n_tri)])
    return out
