"""1D port of un-CNN (Encin et al., 2026, bioRxiv 10.64898/2026.06.07.730652).

Source architecture: https://github.com/epics-lab/untrained-cnn (uncnn.py).
Every structural choice below mirrors that file, with Conv3d/AvgPool3d ->
Conv1d/AvgPool1d. Two variants are provided:

  "faithful"  : kernel 3, readout = adaptive avg-pool to 8 time bins
                (the 1D analogue of the 2x2x2 grid) + covariance pooling.
                Output dim = 9,664, identical to the paper.
  "eeg"       : kernel 15 (at 250 Hz -> deepest receptive field ~1.7 s),
                readout = global mean + std over time + covariance pooling.
                Output dim = 3,904.

The network is NEVER trained. Weights come from PyTorch's default
(Kaiming-uniform) initialisation under a fixed seed and are frozen.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHT_SEED = 0
WIDTHS = (64, 128, 256, 512)


@dataclass(frozen=True)
class VariantConfig:
    kernel_size: int
    readout: str           # "grid8" or "meanstd"
    rank_size: int         # median-filter width for the input channel construction
    cov_channels: int = 32
    groups: int = 8


VARIANTS = {
    "faithful": VariantConfig(kernel_size=3, readout="grid8", rank_size=3),
    "eeg": VariantConfig(kernel_size=15, readout="meanstd", rank_size=5),
}


class DepthwiseSepConv1d(nn.Module):
    """Depthwise (per-channel temporal filter) followed by 1x1 pointwise mixing."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size, padding=pad, groups=in_ch)
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class UnCNN1D(nn.Module):
    """Frozen random-weight 1D CNN feature extractor.

    Input : (batch, in_channels, time). With electrodes-as-channels and three
            signal variants per electrode, in_channels = 3 * n_electrodes.
    Output: (batch, n_features) — see `feature_dim`.
    """

    def __init__(self, in_channels: int, variant: str = "eeg", seed: int | None = WEIGHT_SEED):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {list(VARIANTS)}; got {variant!r}")
        self.variant = variant
        self.cfg = cfg = VARIANTS[variant]
        k, pad, g = cfg.kernel_size, cfg.kernel_size // 2, cfg.groups

        if seed is not None:
            torch.manual_seed(seed)

        # A 1D adaptation: tensor shapes differ from the 3D original, so the
        # same seed does not imply identical numerical weights to that model.
        self.conv1a = nn.Conv1d(in_channels, 64, k, padding=pad)
        self.norm1a = nn.GroupNorm(g, 64)
        self.conv1b = nn.Conv1d(64, 64, k, padding=pad)
        self.norm1b = nn.GroupNorm(g, 64)
        self.down1 = nn.AvgPool1d(2)

        self.conv2a = DepthwiseSepConv1d(64, 128, k)
        self.norm2a = nn.GroupNorm(g, 128)
        self.conv2b = DepthwiseSepConv1d(128, 128, k)
        self.norm2b = nn.GroupNorm(g, 128)
        self.down2 = nn.AvgPool1d(2)

        self.conv3a = DepthwiseSepConv1d(128, 256, k)
        self.norm3a = nn.GroupNorm(g, 256)
        self.conv3b = DepthwiseSepConv1d(256, 256, k)
        self.norm3b = nn.GroupNorm(g, 256)
        self.down3 = nn.AvgPool1d(2)

        self.conv4a = DepthwiseSepConv1d(256, 512, k)
        self.norm4a = nn.GroupNorm(g, 512)
        self.conv4b = DepthwiseSepConv1d(512, 512, k)
        self.norm4b = nn.GroupNorm(g, 512)
        self.down4 = nn.AvgPool1d(2)

        self.grid_pool = nn.AdaptiveAvgPool1d(8)  # 1D analogue of AdaptiveAvgPool3d(2)

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    # ---- readout -----------------------------------------------------------
    def _first_order(self, x):
        b = x.size(0)
        if self.cfg.readout == "grid8":
            return self.grid_pool(x).reshape(b, -1)              # C * 8
        mean = x.mean(dim=2)
        std = x.std(dim=2, unbiased=True)
        return torch.cat([mean, std], dim=1)                     # C * 2

    def _cov_pool(self, x):
        """Upper-triangular channel covariance over time (first `cov_channels`)."""
        b, c = x.size(0), min(x.size(1), self.cfg.cov_channels)
        flat = x[:, :c]
        flat = flat - flat.mean(dim=2, keepdim=True)
        cov = torch.bmm(flat, flat.transpose(1, 2)) / (flat.size(2) - 1)
        idx = torch.triu_indices(c, c, offset=1, device=x.device)
        return cov[:, idx[0], idx[1]]

    def _readout(self, x):
        return torch.cat([self._first_order(x), self._cov_pool(x)], dim=1)

    # ---- forward -----------------------------------------------------------
    def blocks(self, x):
        x1 = F.relu(self.norm1a(self.conv1a(x)))
        x1 = self.down1(F.relu(self.norm1b(self.conv1b(x1))))
        x2 = F.relu(self.norm2a(self.conv2a(x1)))
        x2 = self.down2(F.relu(self.norm2b(self.conv2b(x2))))
        x3 = F.relu(self.norm3a(self.conv3a(x2)))
        x3 = self.down3(F.relu(self.norm3b(self.conv3b(x3))))
        x4 = F.relu(self.norm4a(self.conv4a(x3)))
        x4 = self.down4(F.relu(self.norm4b(self.conv4b(x4))))
        return x1, x2, x3, x4

    @torch.no_grad()
    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != self.conv1a.in_channels or x.shape[2] < 32:
            raise ValueError("Expected (batch, configured channels, time>=32)")
        return torch.cat([self._readout(h) for h in self.blocks(x)], dim=1)

    # ---- bookkeeping ---------------------------------------------------------
    def feature_dim(self) -> int:
        per_ch = 8 if self.cfg.readout == "grid8" else 2
        c = self.cfg.cov_channels
        return sum(WIDTHS) * per_ch + len(WIDTHS) * c * (c - 1) // 2

    def feature_names(self) -> list[str]:
        names = []
        stats = [f"bin{i}" for i in range(8)] if self.cfg.readout == "grid8" else ["mean", "std"]
        c = self.cfg.cov_channels
        for bi, w in enumerate(WIDTHS, start=1):
            if self.cfg.readout == "grid8":
                names += [f"b{bi}_ch{ch}_{s}" for ch in range(w) for s in stats]
            else:
                names += [f"b{bi}_ch{ch}_{s}" for s in stats for ch in range(w)]
            names += [f"b{bi}_cov{i}_{j}" for i in range(c) for j in range(i + 1, c)]
        return names

    def receptive_field(self) -> int:
        """Receptive field (samples) of one unit at the output of block 4."""
        k, rf, jump = self.cfg.kernel_size, 1, 1
        for _ in WIDTHS:
            rf += 2 * (k - 1) * jump   # two convs per block
            rf += (2 - 1) * jump       # avg-pool, kernel 2
            jump *= 2
        return rf


def load_model(in_channels: int, variant: str = "eeg", seed: int = WEIGHT_SEED,
               device: str = "cpu", dtype: torch.dtype = torch.float32) -> UnCNN1D:
    return UnCNN1D(in_channels, variant=variant, seed=seed).to(device=device, dtype=dtype)
