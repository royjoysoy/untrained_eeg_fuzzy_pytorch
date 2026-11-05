"""generic() with the c1 estimator must reproduce untrained_eeg.tasks.probe exactly."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from optimization.evaluate import c1, generic, nested  # noqa: E402

CFG = {"cv_seed": 0, "cv_folds": 5}


def data(task, rng):
    if task == "sleep_stage":
        y = rng.integers(0, 5, 400); g = np.repeat(np.arange(10), 40)
    elif task == "mi_left_right":
        y = np.tile([0, 1], 150); g = np.repeat(np.arange(3), 100)
    elif task == "ec_eo":
        y = np.tile([0, 1], 150); g = np.repeat(np.arange(50), 6)
    else:
        y = np.r_[np.zeros(40, int), np.ones(25, int)]; g = np.arange(65)
    X = rng.standard_normal((len(y), 12)); X[:, 0] += y * 0.8
    return X, y, g


@pytest.mark.parametrize("task", ["bdi_closed", "ec_eo", "sleep_stage", "mi_left_right"])
def test_generic_equals_c1(task):
    X, y, g = data(task, np.random.default_rng(0))
    p1, a1, b1 = c1(X, y, g, task, CFG)
    p2, a2, b2 = generic(X, y, g, task, CFG)
    assert np.allclose(p1, p2, atol=1e-10) and np.isclose(a1, a2) and np.isclose(b1, b2)


@pytest.mark.parametrize("task", ["bdi_closed", "sleep_stage", "mi_left_right"])
def test_nested_prefers_signal(task):
    rng = np.random.default_rng(1)
    X, y, g = data(task, rng)
    X[:, 0] += y * 1.0  # stronger signal for the selection test
    noise = rng.standard_normal(X.shape)
    p, auc, bacc, chosen = nested({"good": (X, ()), "noise": (noise, ())}, y, g, task, CFG)
    assert chosen.count("good") >= len(chosen) - 1 and auc > 0.6


def test_riemann_runs():
    rng = np.random.default_rng(2)
    n, k = 60, 4
    A = rng.standard_normal((n, k, 50))
    C = np.einsum("nkt,nlt->nkl", A, A) / 50
    iu = np.triu_indices(k)
    X = C[:, iu[0], iu[1]]
    y = np.r_[np.zeros(30, int), np.ones(30, int)]
    p, auc, bacc = generic(X, y, np.arange(n), "bdi_closed", CFG, spd_blocks=[[0, k]])
    assert 0 <= auc <= 1


def test_nonfinite_imputed_in_fold():
    rng = np.random.default_rng(3)
    X, y, g = data("bdi_closed", rng)
    X[3, 5] = np.nan; X[10, 2] = np.inf; X[:, 7] = np.nan  # one feature empty everywhere
    p, auc, bacc = generic(X, y, g, "bdi_closed", CFG)
    assert np.isfinite(p).all() and 0 <= auc <= 1
    p2, auc2, _, chosen = nested({"a": (X, ()), "b": (np.nan_to_num(X), ())}, y, g, "bdi_closed", CFG)
    assert np.isfinite(p2).all()


def test_finite_path_unchanged():
    X, y, g = data("bdi_closed", np.random.default_rng(0))
    assert np.array_equal(generic(X, y, g, "bdi_closed", CFG)[0], c1(X, y, g, "bdi_closed", CFG)[0])
