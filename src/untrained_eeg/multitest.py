"""Multiple-testing corrections shared by scripts/permutation.py and scripts/stats.py."""

import numpy as np


def holm(p):
    """Holm-adjusted p-values."""
    p = np.asarray(p, float)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def maxt(null, observed):
    """Westfall-Young step-down min-P adjusted p-values (Westfall and Young 1993, algorithm 2.8).

    null: (n permutations x m tests) null statistics, computed on the same permutations for
    every test; observed: (m,). A larger statistic is stronger evidence. The observed data
    count as one of the n + 1 permutations, so raw p = (1 + #null >= observed) / (1 + n).
    """
    t = np.vstack([observed, null])  # row 0: the observed data
    n1, m = t.shape
    # every row as p-values, each within its own test's column: share of rows >= it
    srt = np.sort(t, axis=0)
    p = np.column_stack([(n1 - np.searchsorted(srt[:, j], t[:, j], "left")) / n1 for j in range(m)])
    raw = p[0]
    order = np.argsort(raw, kind="stable")
    # successive minima from the least significant test up: q[:, r] = min over tests order[r:]
    q = np.minimum.accumulate(p[:, order[::-1]], axis=1)[:, ::-1]
    adj = (q <= raw[order]).sum(axis=0) / n1
    out = np.empty(m)
    out[order] = np.minimum(np.maximum.accumulate(adj), 1.0)  # monotone in the order of raw p
    return out
