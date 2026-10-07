import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uncnn1d import UnCNN1D, build_input  # noqa: E402

N_EL, T = 64, 256  # Small multichannel fixture; real blocks are tested separately.


@pytest.fixture(scope="module")
def x():
    rng = np.random.default_rng(0)
    seg = rng.standard_normal((N_EL, T)).astype(np.float32) * 1e-5
    return torch.from_numpy(build_input(seg)).unsqueeze(0)


def test_input_shape(x):
    assert x.shape == (1, 3 * N_EL, T)
    assert torch.isfinite(x).all()


def test_invalid_input_rejected():
    with pytest.raises(ValueError):
        build_input(np.full((64, 256), np.nan))
    with pytest.raises(ValueError):
        UnCNN1D(192)(torch.zeros(1, 192, 16))


@pytest.mark.parametrize("variant,dim", [("faithful", 9664), ("eeg", 3904)])
def test_feature_dim(x, variant, dim):
    m = UnCNN1D(3 * N_EL, variant=variant, seed=0)
    f = m(x)
    assert f.shape == (1, dim) == (1, m.feature_dim())
    assert len(m.feature_names()) == dim
    assert torch.isfinite(f).all()


def test_frozen_and_deterministic(x):
    a = UnCNN1D(3 * N_EL, variant="eeg", seed=0)
    b = UnCNN1D(3 * N_EL, variant="eeg", seed=0)
    c = UnCNN1D(3 * N_EL, variant="eeg", seed=1)
    assert not any(p.requires_grad for p in a.parameters())
    assert torch.equal(a(x), b(x))
    assert not torch.equal(a(x), c(x))


def test_receptive_field():
    assert UnCNN1D(3, variant="faithful").receptive_field() == 76
    assert UnCNN1D(3, variant="eeg").receptive_field() == 436
