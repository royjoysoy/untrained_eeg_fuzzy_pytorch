"""Random-weight CNNs used as fixed feature extractors. They are never trained.

Each model maps one EEG window (n_channels, n_times) to a flat feature vector.

ShallowFBCSPNet and EEGNet are plain-PyTorch copies of the braindecode 1.8.1
models, without their classifier head. We copy them so this file only needs
torch: the Fuzzy PyTorch environment cannot install braindecode (it pulls in
mne, which needs a newer NumPy than the instrumented PyTorch). For the same
seed they give exactly the same weights and features as braindecode
(checked by tests/test_models.py).
"""

import torch
from torch import nn
from torch.nn import functional as F

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


def _dropped_classifier(in_channels, kernel_size):
    """The classifier conv braindecode builds, which we never use.

    Creating (and initializing) it draws random numbers. Doing the same, in the
    same order, keeps every weight identical to braindecode. Not kept as a module.
    """
    return nn.Conv2d(in_channels, 2, kernel_size)  # 2 outputs (high / low BDI)


class ShallowFBCSPNet(nn.Module):
    """ShallowConvNet (Schirrmeister et al. 2017), braindecode defaults.

    temporal conv -> spatial conv -> batch norm -> square -> mean pool -> log
    """

    def __init__(self, n_chans, n_times, n_filters=40, filter_time_length=25,
                 pool_time_length=75, pool_time_stride=15):
        super().__init__()
        # braindecode's CombinedConv: two convs, merged into one at forward time
        self.conv_time = nn.Conv2d(1, n_filters, (filter_time_length, 1))
        self.conv_spat = nn.Conv2d(n_filters, n_filters, (1, n_chans), bias=False)
        self.bnorm = nn.BatchNorm2d(n_filters, momentum=0.1)
        self.pool = nn.AvgPool2d((pool_time_length, 1), stride=(pool_time_stride, 1))
        self.drop = nn.Dropout(0.5)

        n_out_times = (n_times - filter_time_length + 1 - pool_time_length) // pool_time_stride + 1
        classifier = _dropped_classifier(n_filters, (n_out_times, 1))
        nn.init.xavier_uniform_(self.conv_time.weight, gain=1)
        nn.init.constant_(self.conv_time.bias, 0)
        nn.init.xavier_uniform_(self.conv_spat.weight, gain=1)
        nn.init.xavier_uniform_(classifier.weight, gain=1)

    def forward(self, x):  # x: (batch, n_chans, n_times)
        x = x.unsqueeze(1).transpose(2, 3)  # (batch, 1, n_times, n_chans)
        # temporal and spatial filters combined into one conv, as braindecode does
        weight = (self.conv_time.weight * self.conv_spat.weight.permute(1, 0, 2, 3)).sum(0).unsqueeze(1)
        bias = self.conv_spat.weight.squeeze().sum(-1).mm(self.conv_time.bias.unsqueeze(-1)).squeeze()
        x = F.conv2d(x, weight, bias)
        x = self.bnorm(x)
        x = x * x
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        return self.drop(x)


class Conv2dWithConstraint(nn.Conv2d):
    """Conv2d whose filters are rescaled to an L2 norm of at most `max_norm` at every call."""

    def __init__(self, *args, max_norm=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm
        nn.init.xavier_uniform_(self.weight, gain=1)

    def forward(self, x):
        weight = self.weight.renorm(p=2, dim=0, maxnorm=self.max_norm)
        return self._conv_forward(x, weight, self.bias)


class EEGNet(nn.Module):
    """EEGNet (Lawhern et al. 2018), braindecode defaults: F1=8, D=2, F2=16."""

    def __init__(self, n_chans, n_times, F1=8, D=2, kernel_length=64):
        super().__init__()
        F2 = F1 * D
        bn = dict(momentum=0.01, eps=1e-3)
        self.features = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), bias=False, padding=(0, kernel_length // 2)),
            nn.BatchNorm2d(F1, **bn),
            Conv2dWithConstraint(F1, F1 * D, (n_chans, 1), max_norm=1, bias=False, groups=F1),
            nn.BatchNorm2d(F1 * D, **bn),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.25),
            nn.Conv2d(F1 * D, F1 * D, (1, 16), bias=False, groups=F1 * D, padding=(0, 8)),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2, **bn),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25),
        )
        # braindecode finds the output size with one forward pass on zeros, in
        # training mode. That pass draws dropout random numbers and moves the
        # batch-norm running variance from 1 to 0.99. We repeat it for identical results.
        with torch.inference_mode():
            n_out_times = self.forward(torch.zeros(1, n_chans, n_times)).shape[3]
        _dropped_classifier(F2, (1, n_out_times))
        # braindecode then calls glorot_weight_zero_bias, which (despite its name)
        # only sets batch-norm weights to 1 and biases to 0: already the defaults here.

    def forward(self, x):  # x: (batch, n_chans, n_times)
        return self.features(x.unsqueeze(1))  # (batch, 1, n_chans, n_times)


def build_model(name, n_chans, n_times, seed):
    """Create a model with random weights drawn from `seed`, in eval mode."""
    torch.manual_seed(seed)
    if name == "cnn1d":
        model = SmallCNN1d(n_chans)
    elif name == "shallow":
        model = ShallowFBCSPNet(n_chans, n_times)
    elif name == "eegnet":
        model = EEGNet(n_chans, n_times)
    else:
        raise ValueError(f"unknown model {name!r}, choose from {MODELS}")
    return model.eval()  # eval: no dropout, batch-norm uses its running stats


@torch.no_grad()
def embed(model, X, batch_size=512):
    """Features for every window: (n_windows, n_features) as float64."""
    out = [model(torch.from_numpy(X[i : i + batch_size])).flatten(1)
           for i in range(0, len(X), batch_size)]
    return torch.cat(out).double().numpy()
