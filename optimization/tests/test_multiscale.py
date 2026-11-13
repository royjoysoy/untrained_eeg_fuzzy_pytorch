"""Step (a'') multiscale readout and filter-bank input on synthetic windows (no data needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from untrained_eeg.models import embed  # noqa: E402
from optimization import multiscale, readout  # noqa: E402
from optimization.arch import FB_BANDS, ConfigurableCNN1d, filter_bank, from_name, prepare_input  # noqa: E402
from optimization.multiscale_step0 import DIMS, SAME  # noqa: E402

BASE = from_name(multiscale.BASE)
SF, PER_BLOCK = 128, 3


@pytest.fixture(scope="module")
def X():  # one participant: 2 blocks of 3 windows, 19 electrodes, 2 s at 128 Hz
    return (np.random.default_rng(0).standard_normal((6, 19, 256)) * 10).astype(np.float32)


def feats(level, X, seed=0):
    run = multiscale.arch_for(BASE, level)
    m = ConfigurableCNN1d(X.shape[1], run, seed=seed)
    return embed(m, prepare_input(X, run, SF, PER_BLOCK))


def test_base_is_step_ap_hjorth(X):
    """ref = step (a')'s "hjorth" level of the combined architecture: same config, same numbers."""
    comb = from_name("cov_channels=32,n_filters=256,readout=meanstd_ppv_max")
    assert readout.arch_for(comb, "hjorth", SF) == BASE
    m = ConfigurableCNN1d(X.shape[1], readout.arch_for(comb, "hjorth", SF), seed=0)
    assert np.array_equal(embed(m, X), feats("ref", X))


@pytest.mark.parametrize("level", list(multiscale.LEVELS))
def test_level_runs(X, level):
    f = feats(level, X)
    assert f.shape == (len(X), DIMS[level]) and np.isfinite(f).all()  # float32 once saved


def test_unchanged_columns_are_bit_identical(X):
    """Same weights in the shared blocks -> the same numbers (what step 0 checks on real data)."""
    F = {lv: feats(lv, X) for lv in multiscale.LEVELS}
    for lv, cols, other, ocols in SAME:
        assert np.array_equal(F[lv][:, cols], F[other][:, ocols]), (lv, other)


def test_filter_bank_layout_and_bands(X):
    """Version-major channels; raw copy untouched; each copy's power sits in its band."""
    fb = filter_bank(X, SF, PER_BLOCK)
    n = X.shape[1]
    assert fb.shape == (len(X), 6 * n, X.shape[2]) and fb.dtype == np.float32
    assert np.array_equal(fb[:, :n], X)
    f, p = welch(fb[:, n:].reshape(len(X), 5, n, -1), fs=SF, nperseg=128, axis=-1)
    for b, (lo, hi) in enumerate(FB_BANDS.values()):
        inside = p[:, b][..., (f >= lo) & (f <= hi)].sum()
        assert inside / p[:, b].sum() > 0.8, (lo, hi)


def test_filter_bank_joins_windows_of_a_block(X):
    """Filtering a block's joined windows = filtering the block as one signal (no edge per window)."""
    from scipy.signal import butter, sosfiltfilt
    lo, hi = FB_BANDS["alpha"]
    sos = butter(4, [lo, hi], btype="bandpass", fs=SF, output="sos")
    block = np.concatenate(list(X[:PER_BLOCK]), axis=1).astype(np.float64)
    want = sosfiltfilt(sos, block, axis=-1).astype(np.float32)
    got = filter_bank(X, SF, PER_BLOCK)[:PER_BLOCK, 3 * X.shape[1]: 4 * X.shape[1]]
    assert np.array_equal(np.concatenate(list(got), axis=1), want)


def test_filter_bank_leftover_windows():
    Y = np.random.default_rng(1).standard_normal((7, 2, 256)).astype(np.float32)  # 2 blocks + 1
    assert filter_bank(Y, SF, PER_BLOCK).shape == (7, 12, 256)


def test_filter_bank_needs_rate():
    with pytest.raises(ValueError):
        prepare_input(np.zeros((3, 2, 256), np.float32), multiscale.arch_for(BASE, "fb"))
