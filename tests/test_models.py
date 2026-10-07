"""Our copies of ShallowFBCSPNet / EEGNet must match braindecode 1.8.1 exactly.

    python -m pytest tests/      (needs braindecode; skipped without it)
"""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.models import build_model

braindecode_models = pytest.importorskip("braindecode.models")

N_CHANS, N_TIMES = 60, 256  # 60 channels, 2 s at 128 Hz (configs/ds003478.yaml)


def braindecode_model(name, seed):
    """The model as build_model created it before we copied the code."""
    torch.manual_seed(seed)
    cls = {"shallow": braindecode_models.ShallowFBCSPNet, "eegnet": braindecode_models.EEGNet}[name]
    model = cls(n_chans=N_CHANS, n_outputs=2, n_times=N_TIMES)
    model.final_layer = nn.Identity()
    return model.eval()


def tensors(model):
    """All float weights and buffers (parameter names differ from braindecode's)."""
    return [t for t in model.state_dict().values() if t.dtype.is_floating_point]


@pytest.mark.parametrize("name", ["shallow", "eegnet"])
@pytest.mark.parametrize("seed", [0, 1, 42])
def test_same_as_braindecode(name, seed):
    ours = build_model(name, N_CHANS, N_TIMES, seed)
    ref = braindecode_model(name, seed)
    x = torch.randn(8, N_CHANS, N_TIMES, generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        assert torch.equal(ours(x), ref(x))  # bit-identical features
    # same weights and batch-norm statistics, compared as sorted multisets of values
    a = sorted(t.flatten().tolist() for t in tensors(ours))
    b = sorted(t.flatten().tolist() for t in tensors(ref))
    assert a == b


@pytest.mark.parametrize("name", ["cnn1d", "shallow", "eegnet"])
def test_deterministic(name):
    x = torch.randn(4, N_CHANS, N_TIMES)
    with torch.no_grad():
        a = build_model(name, N_CHANS, N_TIMES, seed=3)(x)
        b = build_model(name, N_CHANS, N_TIMES, seed=3)(x)
        c = build_model(name, N_CHANS, N_TIMES, seed=4)(x)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
