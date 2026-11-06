"""MiniRocket baseline: fitted on training windows only, scored with the shared probe."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
pytest.importorskip("aeon")
from optimization import minirocket as mr  # noqa: E402


def test_fit_sees_training_participants_only(monkeypatch):
    rng = np.random.default_rng(0)
    pids = [f"sub-{i:03d}" for i in range(20)]
    windows = {p: (rng.standard_normal((4, 3, 64)).astype(np.float32) + i % 2,
                   np.array(["closed"] * 4)) for i, p in enumerate(pids)}
    participants = pd.DataFrame({"participant_id": pids})
    B = mr.Blocks(participants, windows, per_block=2)
    labels = pd.DataFrame({"label": [i % 2 for i in range(20)]}, index=pids)
    seen = []
    from aeon.transformations.collection.convolution_based import MiniRocket
    real_fit = MiniRocket.fit

    def spy(self, X, y=None):
        seen.append({p for p in pids if any(np.array_equal(w, x) for w in windows[p][0] for x in X[:5])})
        return real_fit(self, X, y)
    monkeypatch.setattr(MiniRocket, "fit", spy)
    cfg = {"cv_seed": 0, "cv_folds": 5}
    (prob, auc, bacc), dim, n = mr.run_task(B, "bdi_closed", labels, cfg, seed=0, n_jobs=1, max_fit_windows=10_000)
    assert n == 20 and auc > 0.8 and len(seen) == 5
    from untrained_eeg.tasks import _outer_splits
    X0 = np.zeros((20, 1)); y = np.array([i % 2 for i in range(20)])
    for (tr, te), s in zip(_outer_splits(X0, y, None, "bdi_closed", 0, 5), seen):
        assert not s & {pids[i] for i in te}
