"""Default ArchConfig must be phase-1 'cnn1d' (build_model), bit for bit."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from untrained_eeg.models import build_model, embed  # noqa: E402
from optimization.arch import ArchConfig, ConfigurableCNN1d, ablation_grid, from_name, prepare_input  # noqa: E402


@pytest.fixture(scope="module")
def X():
    return (np.random.default_rng(0).standard_normal((6, 60, 256)) * 8).astype(np.float32)


@pytest.mark.parametrize("seed", [0, 7, 29])
def test_default_is_phase1_cnn1d(X, seed):
    ref = build_model("cnn1d", 60, 256, seed)
    new = ConfigurableCNN1d(60, ArchConfig(), seed=seed)
    a = [p for p in ref.parameters()]
    b = [p for p in new.parameters()]
    assert len(a) == len(b) and all(torch.equal(u, v) for u, v in zip(a, b))
    assert np.array_equal(embed(ref, X), embed(new, X))


@pytest.mark.parametrize("n_chans,T", [(60, 256), (19, 256), (2, 3000), (22, 256)])
def test_grid_runs(n_chans, T):
    x = (np.random.default_rng(1).standard_normal((3, n_chans, T)) * 8).astype(np.float32)
    for fac, val, cfg in ablation_grid():
        m = ConfigurableCNN1d(n_chans, cfg, seed=0)
        f = embed(m, prepare_input(x, cfg))
        assert f.shape[0] == 3 and np.isfinite(f).all(), (fac, val)
        assert from_name(cfg.name()) == cfg


def test_per_electrode_no_crosstalk():
    cfg = ArchConfig(channel_mode="per_electrode", readout_blocks="all")
    m = ConfigurableCNN1d(10, cfg, seed=0)
    x = np.random.default_rng(2).standard_normal((1, 10, 256)).astype(np.float32)
    y = x.copy(); y[0, 3] += 1
    fa, fb = embed(m, x)[0], embed(m, y)[0]
    changed = np.flatnonzero(fa != fb)
    # block 1: 1 filter per electrode (cols 0..9), block 2: 2 per electrode (cols 10..29)
    assert set(changed) == {3, 10 + 6, 10 + 7}


def test_receptive_field_default():
    assert ArchConfig().receptive_field() == 68
