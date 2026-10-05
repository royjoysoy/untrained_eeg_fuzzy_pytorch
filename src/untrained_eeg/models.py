"""Random-weight CNNs used as fixed feature extractors. They are never trained.

Each model maps one EEG window (n_channels, n_times) to a flat feature vector.
The classifier head of the braindecode models is removed.
"""

import torch
from torch import nn

MODELS = ("cnn1d", "shallow", "eegnet")


class SmallCNN1d(nn.Module):
    """A tiny 1D CNN: two temporal conv blocks, then average over time."""

    def __init__(self, n_chans, n_filters=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_chans, n_filters, kernel_size=25, padding="same"),
            nn.ELU(),
            nn.AvgPool1d(4),
            nn.Conv1d(n_filters, 2 * n_filters, kernel_size=11, padding="same"),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),  # one value per filter
        )

    def forward(self, x):  # x: (batch, n_chans, n_times)
        return self.net(x)


def build_model(name, n_chans, n_times, seed):
    """Create a model with random weights drawn from `seed`, in eval mode."""
    from braindecode.models import EEGNet, ShallowFBCSPNet

    torch.manual_seed(seed)
    if name == "cnn1d":
        model = SmallCNN1d(n_chans)
    elif name == "shallow":
        model = ShallowFBCSPNet(n_chans=n_chans, n_outputs=2, n_times=n_times)
        model.final_layer = nn.Identity()  # drop the classifier
    elif name == "eegnet":
        model = EEGNet(n_chans=n_chans, n_outputs=2, n_times=n_times)
        model.final_layer = nn.Identity()
    else:
        raise ValueError(f"unknown model {name!r}, choose from {MODELS}")
    return model.eval()  # eval: no dropout, batch-norm uses its default stats


@torch.no_grad()
def embed(model, X, batch_size=512):
    """Features for every window: (n_windows, n_features) as float64."""
    out = [model(torch.from_numpy(X[i : i + batch_size])).flatten(1)
           for i in range(0, len(X), batch_size)]
    return torch.cat(out).double().numpy()
