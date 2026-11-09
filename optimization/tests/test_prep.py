"""Step (b) preprocessing levels on synthetic windows (no data needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from optimization import prep  # noqa: E402
from optimization.arch import ArchConfig  # noqa: E402
from optimization.common import block_features, dataset_config  # noqa: E402

SF, PER_BLOCK, T = 128, 30, 256  # 60 s blocks of 2 s windows at 128 Hz
CFG = {"sfreq": SF, "segment_s": 60, "window_s": 2}


@pytest.fixture
def windows():
    rng = np.random.default_rng(0)
    return {p: ((rng.standard_normal((2 * PER_BLOCK, 4, T)) * 10).astype(np.float32),
                np.array(["closed"] * PER_BLOCK + ["open"] * PER_BLOCK)) for p in ("sub-1", "sub-2")}


def test_reference_is_untouched(windows):
    for lv in ("ref", "sf256", "sf256_k2", "bp05_45"):  # these change the cache read, not the windows
        W, b, keep = prep.transform(windows, PER_BLOCK, CFG, prep.LEVELS[lv])
        assert W is windows and b == PER_BLOCK and keep is None


def test_longer_windows_are_joined_windows(windows):
    W, b, _ = prep.transform(windows, PER_BLOCK, CFG, prep.LEVELS["win10"])
    X, cond = windows["sub-1"]
    assert b == 6 and W["sub-1"][0].shape == (12, 4, 5 * T)
    assert np.array_equal(W["sub-1"][0][1], np.concatenate(list(X[5:10]), axis=1))
    assert list(W["sub-1"][1]) == ["closed"] * 6 + ["open"] * 6


def test_normalization(windows):
    W, _, _ = prep.transform(windows, PER_BLOCK, CFG, prep.LEVELS["norm_subject"])
    X = W["sub-1"][0].astype(np.float64)
    assert np.allclose(X.mean(axis=(0, 2)), 0, atol=1e-5) and np.allclose(X.std(axis=(0, 2)), 1, atol=1e-4)
    W, _, _ = prep.transform(windows, PER_BLOCK, CFG, prep.LEVELS["norm_window"])
    assert np.allclose(W["sub-1"][0].std(axis=2), 1, atol=1e-4)


def test_lowpass_removes_high_frequencies():
    t = np.arange(PER_BLOCK * T) / SF
    x = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 38 * t)
    X = x.reshape(PER_BLOCK, T)[:, None, :].astype(np.float32)
    Y = prep.lowpass_blocks(X, PER_BLOCK, SF, 30.0).reshape(-1)
    spec = np.abs(np.fft.rfft(Y[SF * 5:-SF * 5]))  # away from the block edges
    f = np.fft.rfftfreq(len(Y) - SF * 10, 1 / SF)
    assert spec[np.argmin(abs(f - 38))] < 0.05 * spec[np.argmin(abs(f - 10))]


def test_artifact_rejection_keeps_rows(windows):
    X = np.concatenate([windows["sub-1"][0], windows["sub-2"][0]])  # 4 blocks
    cond = np.concatenate([windows["sub-1"][1], windows["sub-2"][1]])
    X[3, 2, 10] = 1000                  # one bad window in block 0
    X[PER_BLOCK:2 * PER_BLOCK, 0, 5] = 1000  # every window of block 1 is bad -> it keeps them all
    w = {"sub-1": (X, cond)}
    for lv in ("rej_p2p150", "rej_mad5"):
        _, _, keep = prep.transform(w, PER_BLOCK, CFG, prep.LEVELS[lv])
        k = keep["sub-1"]
        assert not k[3] and k[:PER_BLOCK].sum() == PER_BLOCK - 1 and k[PER_BLOCK:].all(), lv
        F = np.arange(len(X), dtype=np.float32)[:, None]
        B, c = block_features(F, cond, PER_BLOCK, k)
        assert len(B) == 4 and list(c) == ["closed", "open"] * 2
        assert B[0, 0] == np.delete(F[:PER_BLOCK, 0], 3).mean()


def test_aggregation():
    F = np.random.default_rng(1).standard_normal((60, 5)).astype(np.float32)
    cond = np.array(["a"] * 30 + ["b"] * 30)
    ref, _ = block_features(F, cond, 30)
    med, _ = block_features(F, cond, 30, agg="median")
    mean2, _ = block_features(F, cond, 30, keep=np.ones(60, bool))
    assert np.allclose(mean2, ref) and med.dtype == np.float32
    assert np.allclose(med[1], np.median(F[30:], axis=0))
    tr, _ = block_features(F, cond, 30, agg="trim20")
    s = np.sort(F[:30], axis=0)[6:24]
    assert np.allclose(tr[0], s.mean(axis=0), atol=1e-6)


def test_kernel_x2_keeps_time_spans():
    a = ArchConfig()
    b = prep.arch_for(a, "sf256_k2")
    assert (b.kernel1, b.pool) == (51, 8) and prep.arch_for(a, "sf256") == a
    assert abs(b.receptive_field() / 256 - a.receptive_field() / 128) < 0.02  # seconds


def test_where_levels_apply():
    got = {ds: {lv for lv in prep.LEVELS if prep.applies(lv, dataset_config(ds, None))}
           for ds in ("ds003478", "lemon", "sleep_edf", "bnci2014_001")}
    every = set(prep.LEVELS)
    assert got["ds003478"] == got["lemon"] == every
    assert got["sleep_edf"] == {"ref", "norm_subject", "norm_window", "lp30", "bp05_45"}
    assert got["bnci2014_001"] == every - {"win10", "rej_p2p150", "rej_mad5", "agg_median", "agg_trim20"}
