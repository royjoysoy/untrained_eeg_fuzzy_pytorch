"""Configurable frozen random-weight 1D CNN for the architecture/initialisation ablation.

`ArchConfig()` with all defaults is the phase-1 "cnn1d" extractor
(src/untrained_eeg/models.py: SmallCNN1d, built by build_model("cnn1d", ...)):

    Conv1d(n_chans, 32, k=25, 'same') -> ELU -> AvgPool(4)
    -> Conv1d(32, 64, k=11, 'same') -> ELU -> mean over time      (64 features)

For the same seed it draws the same weights in the same order and returns
bit-identical features (optimization/tests/test_arch.py). Every other field
changes ONE factor away from that default. Weights are never trained.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ArchConfig:
    # a. channel handling
    channel_mode: str = "mixed"        # mixed | depthwise_first | per_electrode
    # b. input versions
    input_versions: str = "raw"        # raw | unc3 (robust-scaled, median-filtered, |derivative|)
    # c. kernels / dilation
    kernel1: int = 25                  # first conv (25 samples = 0.2 s at 128 Hz)
    kernel_rest: int = 11              # every later conv
    dilation: int = 1                  # all convs
    # d. multi-scale first layer: kernels of parallel branches, or ()
    multiscale: tuple = ()
    # e. depth and width
    n_blocks: int = 2                  # conv blocks; AvgPool(4) between blocks
    n_filters: int = 32                # first block width; doubles every block
    # f. initialisation
    init: str = "torch_default"        # torch_default | kaiming_normal | xavier_uniform | gaussian
    weight_scale: float = 1.0
    bias: str = "default"              # default | zero
    # g. nonlinearity and readout
    nonlinearity: str = "elu"          # elu | relu | tanh | none
    readout: str = "mean"              # mean | meanstd | meanstd_ppv_max
    readout_blocks: str = "last"       # last | all
    cov_channels: object = 0           # 0, an int, or "all": channel covariance of the readout block(s)

    def name(self) -> str:
        d = ArchConfig()
        diff = {k: v for k, v in asdict(self).items() if v != getattr(d, k)}
        return "default" if not diff else ",".join(f"{k}={_fmt(v)}" for k, v in sorted(diff.items()))

    def receptive_field(self) -> int:
        """Samples seen by one unit of the last conv (before the global readout)."""
        k1 = max(self.multiscale) if self.multiscale else self.kernel1
        rf, jump = 1 + (k1 - 1) * self.dilation, 1
        for _ in range(1, self.n_blocks):
            jump *= 4
            rf += 3 * (jump // 4) + (self.kernel_rest - 1) * self.dilation * jump
        return rf


def _fmt(v):
    return "-".join(map(str, v)) if isinstance(v, tuple) else str(v)


def from_name(name: str) -> ArchConfig:
    """Inverse of ArchConfig.name()."""
    if name == "default":
        return ArchConfig()
    d, kw = ArchConfig(), {}
    for part in name.split(","):
        k, v = part.split("=", 1)
        ref = getattr(d, k)
        if k == "multiscale":
            kw[k] = tuple(int(x) for x in v.split("-")) if v else ()
        elif k == "cov_channels":
            kw[k] = v if v == "all" else int(v)
        else:
            kw[k] = type(ref)(v)
    return ArchConfig(**kw)


# ---------------------------------------------------------------------------
def _conv(cin, cout, k, d, groups=1):
    return nn.Conv1d(cin, cout, k, padding="same", dilation=d, groups=groups)


class MultiScale(nn.Module):
    def __init__(self, cin, cout, kernels, groups=1):
        super().__init__()
        n = len(kernels)
        sizes = [cout // n + (i < cout % n) for i in range(n)]
        if groups > 1:
            sizes = [s - s % groups or groups for s in sizes]
        self.branches = nn.ModuleList(_conv(cin, s, k, 1, groups) for s, k in zip(sizes, kernels))
        self.out_channels = sum(sizes)

    def forward(self, x):
        return torch.cat([b(x) for b in self.branches], dim=1)


class DepthwiseFirst(nn.Module):
    """Temporal filter per input channel, then 1x1 spatial mixing (EEGNet-like separation)."""

    def __init__(self, cin, cout, k, d):
        super().__init__()
        self.depthwise = _conv(cin, cin, k, d, groups=cin)
        self.pointwise = nn.Conv1d(cin, cout, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ConfigurableCNN1d(nn.Module):
    """n_chans: electrodes in the input (before the unc3 expansion)."""

    def __init__(self, n_chans: int, cfg: ArchConfig = ArchConfig(), seed: int | None = 0):
        super().__init__()
        self.cfg, self.n_el = cfg, n_chans
        n_ver = 3 if cfg.input_versions == "unc3" else 1
        cin = n_chans * n_ver
        per_el = cfg.channel_mode == "per_electrode"
        g = n_chans if per_el else 1
        if per_el and n_ver > 1:  # electrode-major order so each group holds one electrode's versions
            perm = [e + v * n_chans for e in range(n_chans) for v in range(n_ver)]
            self.register_buffer("perm", torch.tensor(perm), persistent=False)
        else:
            self.perm = None
        # widths: n_filters * 2**b, or (per electrode) 1, 2, 4, ... filters per electrode
        widths = [n_chans * 2 ** b if per_el else cfg.n_filters * 2 ** b for b in range(cfg.n_blocks)]

        if seed is not None:
            torch.manual_seed(seed)  # same call as untrained_eeg.models.build_model
        self.convs = nn.ModuleList()
        prev = cin
        for b, w in enumerate(widths):
            if b == 0 and cfg.multiscale:
                conv = MultiScale(prev, w, cfg.multiscale, groups=g)
                w = conv.out_channels
            elif b == 0 and cfg.channel_mode == "depthwise_first":
                conv = DepthwiseFirst(prev, w, cfg.kernel1, cfg.dilation)
            else:
                conv = _conv(prev, w, cfg.kernel1 if b == 0 else cfg.kernel_rest, cfg.dilation, g)
            self.convs.append(conv)
            prev = w
        self.widths = widths
        self._reinit()
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def _reinit(self):
        c = self.cfg
        if c.init == "torch_default" and c.weight_scale == 1.0 and c.bias == "default":
            return  # keep PyTorch's draw untouched (bit-identical default)
        for m in self.convs.modules():
            if not isinstance(m, nn.Conv1d):
                continue
            if c.init == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif c.init == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight)
            elif c.init == "gaussian":
                nn.init.normal_(m.weight, 0.0, 1.0)
            elif c.init != "torch_default":
                raise ValueError(c.init)
            m.weight.mul_(c.weight_scale)
            if c.bias == "zero":
                m.bias.zero_()
            elif c.bias != "default":
                raise ValueError(c.bias)

    def _act(self, x):
        nl = self.cfg.nonlinearity
        if nl == "elu":
            return F.elu(x)
        if nl == "relu":
            return F.relu(x)
        if nl == "tanh":
            return torch.tanh(x)
        if nl == "none":
            return x
        raise ValueError(nl)

    def _readout(self, h):
        c = self.cfg
        parts = [h.mean(dim=2)]
        if c.readout in ("meanstd", "meanstd_ppv_max"):
            parts.append(h.std(dim=2, unbiased=True))
        if c.readout == "meanstd_ppv_max":
            parts += [(h > 0).float().mean(dim=2), h.amax(dim=2)]
        elif c.readout not in ("mean", "meanstd"):
            raise ValueError(c.readout)
        n = 0 if not c.cov_channels else h.size(1) if c.cov_channels == "all" else min(int(c.cov_channels), h.size(1))
        if n >= 2:
            z = h[:, :n] - h[:, :n].mean(dim=2, keepdim=True)
            cov = torch.bmm(z, z.transpose(1, 2)) / max(z.size(2) - 1, 1)
            iu = torch.triu_indices(n, n, offset=1, device=h.device)
            parts.append(cov[:, iu[0], iu[1]])
        return torch.cat(parts, dim=1)

    @torch.no_grad()
    def forward(self, x):
        if self.perm is not None:
            x = x[:, self.perm]
        outs = []
        for b, conv in enumerate(self.convs):
            if b > 0:
                x = F.avg_pool1d(x, 4)
            x = self._act(conv(x))
            outs.append(x)
        keep = outs if self.cfg.readout_blocks == "all" else outs[-1:]
        return torch.cat([self._readout(h) for h in keep], dim=1)


def prepare_input(X: np.ndarray, cfg: ArchConfig) -> np.ndarray:
    """(n_windows, n_chans, T) float32 -> the network input for this config."""
    if cfg.input_versions == "raw":
        return X
    if cfg.input_versions == "unc3":
        from uncnn1d.inputs import build_input
        return np.stack([build_input(w, rank_size=5) for w in X])
    raise ValueError(cfg.input_versions)


# ---------------------------------------------------------------------------
def ablation_grid() -> list[tuple[str, str, ArchConfig]]:
    """One factor at a time away from the phase-1 cnn1d."""
    D = ArchConfig()
    g = [("default", "current", D)]

    def add(fac, val, **kw):
        g.append((fac, val, replace(D, **kw)))

    for m in ("depthwise_first", "per_electrode"):
        add("a_channel_mode", m, channel_mode=m)
    add("b_input_versions", "unc3", input_versions="unc3")
    for k in (7, 13, 51, 101):
        add("c_kernel1", f"k1={k}", kernel1=k)
    for dl in (2, 4):
        add("c_dilation", f"d={dl}", dilation=dl)
    add("d_multiscale", "7-25-51-101", multiscale=(7, 25, 51, 101))
    add("d_multiscale", "13-51", multiscale=(13, 51))
    for nb in (1, 3, 4):
        add("e_depth", f"{nb} blocks", n_blocks=nb)
    for nf in (16, 64, 128):
        add("e_width", f"{nf} filters", n_filters=nf)
    for ini in ("kaiming_normal", "xavier_uniform", "gaussian"):
        add("f_init", ini, init=ini)
    for s in (0.1, 10.0):
        add("f_weight_scale", f"x{s:g}", weight_scale=s)
    add("f_bias", "zero", bias="zero")
    for nl in ("relu", "tanh", "none"):
        add("g_nonlinearity", nl, nonlinearity=nl)
    for r in ("meanstd", "meanstd_ppv_max"):
        add("g_readout", r, readout=r)
    add("g_readout_blocks", "all", readout_blocks="all")
    add("g_cov", "32 channels", cov_channels=32)
    return g
