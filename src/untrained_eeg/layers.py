"""Each model as a list of consecutive stages, to perturb one layer at a time (#31, task 6).

stages(model) returns [(name, module), ...] such that applying the modules in order gives
exactly model(x). Dropout layers are left out (identity in eval mode).
"""

import torch
from torch import nn
from torch.nn import functional as F

from .models import EEGNet, ShallowFBCSPNet, SmallCNN1d


class _ShallowConv(nn.Module):
    """ShallowFBCSPNet's temporal + spatial convolution, merged into one, as in its forward."""

    def __init__(self, model):
        super().__init__()
        self.conv_time, self.conv_spat = model.conv_time, model.conv_spat

    def forward(self, x):
        x = x.unsqueeze(1).transpose(2, 3)
        weight = (self.conv_time.weight * self.conv_spat.weight.permute(1, 0, 2, 3)).sum(0).unsqueeze(1)
        bias = self.conv_spat.weight.squeeze().sum(-1).mm(self.conv_time.bias.unsqueeze(-1)).squeeze()
        return F.conv2d(x, weight, bias)


class _Square(nn.Module):
    def forward(self, x):
        return x * x


class _SafeLog(nn.Module):
    def forward(self, x):
        return torch.log(torch.clamp(x, min=1e-6))


class _Unsqueeze(nn.Module):
    def forward(self, x):
        return x.unsqueeze(1)


def stages(model):
    if isinstance(model, SmallCNN1d):
        names = ["conv1", "elu1", "pool1", "conv2", "elu2", "global_pool"]
        return list(zip(names, model.net))
    if isinstance(model, ShallowFBCSPNet):
        return [("conv_time_spat", _ShallowConv(model)), ("batch_norm", model.bnorm), ("square", _Square()),
                ("pool", model.pool), ("log", _SafeLog())]
    if isinstance(model, EEGNet):
        names = ["conv_temporal", "bn1", "conv_spatial_depthwise", "bn2", "elu1", "pool1", None,
                 "conv_separable_depth", "conv_separable_point", "bn3", "elu2", "pool2", None]
        layers = [(n, m) for n, m in zip(names, model.features) if n is not None]  # None: dropout
        first_name, first = layers[0]
        return [(first_name, nn.Sequential(_Unsqueeze(), first))] + layers[1:]
    raise TypeError(f"no stages for {type(model).__name__}")


def check(model, x):
    """Max |stages applied in order - model(x)|: should be 0."""
    with torch.no_grad():
        y = x
        for _, m in stages(model):
            y = m(y)
        return (y - model(x)).abs().max().item()
