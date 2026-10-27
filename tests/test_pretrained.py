"""Pretrained EEGNetv4 checkpoints loaded into our EEGNet must match braindecode 1.8.1 exactly.

    bash scripts/00_download_pretrained.sh && python -m pytest tests/
    (needs braindecode; skipped without it or without the downloaded weights)
"""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.pretrained import (EEGNETV4_CHANNELS, EEGNETV4_DATASETS, PRETRAINED_DIR, SHALLOW_CHANNELS,
                                      SHALLOW_CHECKPOINTS, load_eegnetv4, load_shallow)

braindecode_models = pytest.importorskip("braindecode.models")

N_TIMES = 256  # 2 s at 128 Hz, our window (the checkpoints were trained on 385 samples)


def braindecode_eegnet(state):
    """The checkpoint in braindecode 1.8.1's EEGNet, which renamed two of EEGNetv4's tensors."""
    n_outputs = state["conv_classifier.weight"].shape[0]
    model = braindecode_models.EEGNet(n_chans=len(EEGNETV4_CHANNELS), n_outputs=n_outputs, n_times=385)
    renamed = {"conv_spatial.weight": "conv_spatial.parametrizations.weight.original",
               "conv_classifier.weight": "final_layer.conv_classifier.weight",
               "conv_classifier.bias": "final_layer.conv_classifier.bias"}
    model.load_state_dict({renamed.get(k, k): v for k, v in state.items()}, strict=True)
    model.final_layer = nn.Identity()  # features only, as in our models
    return model.eval()


@pytest.mark.parametrize("dataset", EEGNETV4_DATASETS)
def test_same_as_braindecode(dataset):
    path = PRETRAINED_DIR / "eegnetv4" / dataset / "model-params.pkl"
    if not path.exists():
        pytest.skip("run scripts/00_download_pretrained.sh first")
    state = torch.load(path, map_location="cpu", weights_only=True)
    ours = load_eegnetv4(dataset, N_TIMES)
    ref = braindecode_eegnet(state)
    x = 10 * torch.randn(8, len(EEGNETV4_CHANNELS), N_TIMES, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.equal(ours(x), ref(x))  # bit-identical features


@pytest.mark.parametrize("checkpoint", SHALLOW_CHECKPOINTS)
def test_shallow_same_as_braindecode(checkpoint):
    path = PRETRAINED_DIR / "shallow" / checkpoint / "params.safetensors"
    if not path.exists():
        pytest.skip("run scripts/00_download_pretrained.sh first")
    from safetensors.torch import load_file

    state = load_file(path)
    n_out = state["final_layer.conv_classifier.weight"].shape[2]  # classifier length in time
    ref = braindecode_models.ShallowFBCSPNet(n_chans=len(SHALLOW_CHANNELS), n_outputs=4, n_times=1125,
                                             final_conv_length=n_out)
    ref.load_state_dict(state, strict=True)
    ref.final_layer = nn.Identity()
    ref.eval()
    ours = load_shallow(checkpoint, 1125)  # 4.5 s at 250 Hz, the tutorials' trial length
    x = torch.randn(8, len(SHALLOW_CHANNELS), 1125, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.equal(ours(x), ref(x))
