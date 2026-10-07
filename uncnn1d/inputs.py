"""Three-variant input construction — 1D analogue of un-CNN's 3-channel input.

un-CNN (3D)                         -> here (1D, per electrode, along time)
  scale_robust (clip 2/98, median/IQR) -> same, per electrode
  rank_filter (3D median, size 3)      -> 1D median filter along time
  sobel_mag  (3D gradient magnitude)   -> |central difference| along time

Normalisation of the rank and edge variants (global min-max over the whole
segment) follows uncnn.py. The derivative is computed per electrode only:
scipy's N-d Sobel would also smooth across electrodes, which has no meaning
for a channel axis.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi


def scale_robust(x: np.ndarray, clip_low: float = 2, clip_high: float = 98,
                 eps: float = 1e-8) -> np.ndarray:
    """Per-electrode robust scaling. x: (n_electrodes, n_times)."""
    lo, hi = np.percentile(x, [clip_low, clip_high], axis=1, keepdims=True)
    x = np.clip(x, lo, hi)
    med = np.median(x, axis=1, keepdims=True)
    q25, q75 = np.percentile(x, [25, 75], axis=1, keepdims=True)
    return ((x - med) / (q75 - q25 + eps)).astype(np.float32)


def _minmax(x: np.ndarray) -> np.ndarray:
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def rank_filter(x: np.ndarray, size: int = 3) -> np.ndarray:
    """Local median along time for each electrode, then global min-max."""
    return _minmax(ndi.median_filter(x, size=(1, size), mode="reflect"))


def edge_magnitude(x: np.ndarray) -> np.ndarray:
    """|x[t+1] - x[t-1]| per electrode, then global min-max (1D Sobel analogue)."""
    d = ndi.correlate1d(x, weights=[-1.0, 0.0, 1.0], axis=1, mode="reflect")
    return _minmax(np.abs(d))


def build_input(segment: np.ndarray, rank_size: int = 3) -> np.ndarray:
    """(n_electrodes, n_times) -> (3 * n_electrodes, n_times), float32.

    Channel order: [robust(all electrodes), rank(all), edge(all)].
    """
    if segment.ndim != 2:
        raise ValueError(f"expected (electrodes, times), got shape {segment.shape}")
    if segment.shape[0] == 0 or segment.shape[1] < 32 or not np.isfinite(segment).all():
        raise ValueError("Input must have electrodes, at least 32 time points, and finite values")
    if rank_size <= 0 or rank_size % 2 != 1:
        raise ValueError("Median-filter width must be positive and odd")
    robust = scale_robust(segment)
    ranked = rank_filter(robust, size=rank_size)
    edges = edge_magnitude(robust)
    return np.concatenate([robust, ranked, edges], axis=0).astype(np.float32)
