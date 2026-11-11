"""Step (a') readout levels on synthetic windows (no data needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from untrained_eeg.models import embed  # noqa: E402
from optimization import readout  # noqa: E402
from optimization.arch import ArchConfig, ConfigurableCNN1d, from_name  # noqa: E402
from optimization.common import block_features  # noqa: E402
from optimization.evaluate import RiemannTangent  # noqa: E402

COMBINED = from_name("cov_channels=32,n_filters=256,readout=meanstd_ppv_max")
ARCHS = {"default": ArchConfig(), "combined": COMBINED}
# expected feature dimension per level: (default, combined)
DIMS = {"ref": (64, 2544), "logvar": (64, 512), "logpow_first": (32, 256), "ref_logvar": (128, 3056),
        "subwin": (128, 1024), "across_win": (64, 512),  # x2 after the block's mean_sd
        "hjorth": (192, 1536), "tcov32": (528, 528), "ref_tcov32": (592, 2576), "tcov16": (136, 136)}


def feats(arch, level, X, sfreq=128, seed=0):
    run = readout.arch_for(arch, level, sfreq)
    m = ConfigurableCNN1d(X.shape[1], run, seed=seed)
    return m, embed(m, X)


@pytest.fixture(scope="module")
def X():
    return (np.random.default_rng(0).standard_normal((4, 19, 256)) * 10).astype(np.float32)


@pytest.mark.parametrize("name", list(ARCHS))
@pytest.mark.parametrize("level", list(readout.LEVELS))
def test_level_runs(X, name, level):
    arch = ARCHS[name]
    m, f = feats(arch, level, X)
    assert f.shape == (4, DIMS[level][name == "combined"]) and np.isfinite(f).all()
    run = readout.arch_for(arch, level, 128)
    assert from_name(run.name()) == run  # names round-trip (they label the outputs)
    spd = m.spd_blocks(f.shape[1])
    assert bool(spd) == level.startswith(("tcov", "ref_tcov"))


def test_ref_is_the_architecture_itself():
    for arch in ARCHS.values():
        assert readout.arch_for(arch, "ref", 128) is arch


@pytest.mark.parametrize("name", list(ARCHS))
def test_untouched_columns_are_bit_identical(X, name):
    """Levels that ADD to the readout keep the reference's own columns exactly."""
    arch = ARCHS[name]
    _, ref = feats(arch, "ref", X)
    _, add = feats(arch, "ref_logvar", X)
    head = 64 if name == "default" else 2048  # columns before the covariance
    assert np.array_equal(add[:, :head], ref[:, :head])
    assert np.array_equal(add[:, add.shape[1] - (ref.shape[1] - head):], ref[:, head:])
    _, tc = feats(arch, "ref_tcov32", X)
    assert np.array_equal(tc[:, :head], ref[:, :head])


def test_same_weights_at_every_level():
    a = ConfigurableCNN1d(19, ArchConfig(), seed=3)
    b = ConfigurableCNN1d(19, readout.arch_for(ArchConfig(), "logpow_first", 128), seed=3)
    assert all(torch.equal(u, v) for u, v in zip(a.parameters(), b.parameters()))


def test_logvar_is_the_covariance_diagonal(X):
    """A1 and C1 look at the same filters: unshrunk diagonal = exp(log-variance)."""
    _, lv = feats(ArchConfig(), "logvar", X)
    run = readout.arch_for(ArchConfig(), "tcov32", 128)
    from dataclasses import replace
    m = ConfigurableCNN1d(19, replace(run, shrink=0.0), seed=0)
    c = embed(m, X)
    iu = np.triu_indices(32)
    diag = c[:, iu[0] == iu[1]]
    assert np.allclose(diag, np.exp(lv[:, :32]), rtol=1e-4)


def test_shrinkage(X):
    from dataclasses import replace
    run = readout.arch_for(ArchConfig(), "tcov16", 128)
    raw = embed(ConfigurableCNN1d(19, replace(run, shrink=0.0), seed=0), X)
    shr = embed(ConfigurableCNN1d(19, run, seed=0), X)
    M0, M1 = RiemannTangent._mats(raw, 0, 16), RiemannTangent._mats(shr, 0, 16)
    tr = np.trace(M0, axis1=1, axis2=2)[:, None, None] / 16
    assert np.allclose(M1, 0.9 * M0 + 0.1 * tr * np.eye(16), rtol=1e-4, atol=1e-6)
    assert (np.linalg.eigvalsh(M1).min(axis=1) > 0).all()


def test_logpow_first_is_shallowconvnet_like(X):
    m, f = feats(ArchConfig(), "logpow_first", X)
    with torch.no_grad():
        h = m.convs[0](torch.from_numpy(X))  # first conv, no ELU
    assert np.allclose(f, torch.log(h.pow(2).mean(dim=2)).double().numpy(), rtol=1e-5)


def test_subwin_length():
    assert readout.arch_for(ArchConfig(), "subwin", 128).subwin == 16   # 0.5 s at 128/4 Hz
    assert readout.arch_for(ArchConfig(), "subwin", 100).subwin == 12   # Sleep-EDF, 25 Hz
    x = (np.random.default_rng(1).standard_normal((2, 2, 3000)) * 10).astype(np.float32)
    _, f = feats(ArchConfig(), "subwin", x, sfreq=100)
    assert f.shape == (2, 128) and np.isfinite(f).all()


def test_across_window_mean_sd():
    F = np.random.default_rng(2).standard_normal((60, 5))
    cond = np.array(["closed"] * 30 + ["open"] * 30)
    B, c = block_features(F, cond, 30, agg="mean_sd")
    assert B.shape == (2, 10) and list(c) == ["closed", "open"]
    assert np.allclose(B[1, :5], F[30:].mean(0)) and np.allclose(B[1, 5:], F[30:].std(0, ddof=1))


def test_applies():
    rest = {"segment_s": 60, "window_s": 2}
    bnci = {"segment_s": 4, "window_s": 2}
    sleep = {"segment_s": 30, "window_s": 30}
    assert readout.applies("across_win", rest)
    assert not readout.applies("across_win", bnci) and not readout.applies("across_win", sleep)
    assert all(readout.applies(lv, bnci) for lv in readout.LEVELS if lv != "across_win")


def test_tangent_space_fits_on_block_means(X):
    """Block means of window covariances stay SPD, and RiemannTangent accepts the stored columns."""
    m, f = feats(COMBINED, "ref_tcov32", (np.random.default_rng(3).standard_normal((40, 19, 256)) * 10)
                 .astype(np.float32))
    B, _ = block_features(f, np.array(["a"] * 40), 10)
    spd = m.spd_blocks(f.shape[1])
    Z = RiemannTangent(spd).fit(B.astype(np.float32)).transform(B.astype(np.float32))
    assert Z.shape == (4, 2048 + 528) and np.isfinite(Z).all()
