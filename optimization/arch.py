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
    #                                    | fb6 (raw + 5 band-passed copies, step a'', FB_BANDS)
    # c. kernels / dilation
    kernel1: int = 25                  # first conv (25 samples = 0.2 s at 128 Hz)
    kernel_rest: int = 11              # every later conv
    dilation: int = 1                  # all convs
    # d. multi-scale first layer: kernels of parallel branches, or ()
    multiscale: tuple = ()
    # e. depth and width
    n_blocks: int = 2                  # conv blocks; AvgPool(pool) between blocks
    pool: int = 4                      # 8 at 256 Hz keeps the later layers' time spans as at 128 Hz
    n_filters: int = 32                # first block width; doubles every block
    # f. initialisation
    init: str = "torch_default"        # torch_default | kaiming_normal | xavier_uniform | gaussian
    weight_scale: float = 1.0
    bias: str = "default"              # default | zero
    # g. nonlinearity and readout
    nonlinearity: str = "elu"          # elu | relu | tanh | none
    readout: str = "mean"              # mean | meanstd | meanstd_ppv_max   (steps a, b)
    #                                    logvar | logpow | subwin | hjorth | none   (step a', readout.py)
    readout_blocks: str = "last"       # last | all
    cov_channels: object = 0           # 0, an int, or "all": covariance between the first n FILTER outputs
    # step (a'): readouts (optimization/readout.py); defaults = steps (a) and (b), bit for bit
    readout_from: str = "act"          # act: after the nonlinearity | first_lin: first conv, no nonlinearity
    readout_extra: str = ""            # "" | logvar: log-variance appended to the readout above
    cov_mode: str = "offdiag"          # offdiag: upper triangle without diagonal | spd: with diagonal
    shrink: float = 0.0                # spd only: (1 - a) C + a tr(C)/n I (per window, no labels)
    subwin: int = 0                    # subwin only: sub-window length, in samples of the readout layer

    def name(self) -> str:
        d = ArchConfig()
        diff = {k: v for k, v in asdict(self).items() if v != getattr(d, k)}
        return "default" if not diff else ",".join(f"{k}={_fmt(v)}" for k, v in sorted(diff.items()))

    def receptive_field(self) -> int:
        """Samples seen by one unit of the last conv (before the global readout)."""
        k1 = max(self.multiscale) if self.multiscale else self.kernel1
        rf, jump = 1 + (k1 - 1) * self.dilation, 1
        for _ in range(1, self.n_blocks):
            jump *= self.pool
            rf += (self.pool - 1) * (jump // self.pool) + (self.kernel_rest - 1) * self.dilation * jump
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
_EPS = 1e-6    # floor before a log (ShallowConvNet's safe_log uses the same idea)
_TINY = 1e-12  # floor of a denominator (a constant filter output would give 0/0)


def _log(v):
    return torch.log(v.clamp_min(_EPS))


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
        n_ver = N_VERSIONS[cfg.input_versions]
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

    def _cov_n(self, n_filters):
        """Filters in the covariance readout (0 = none)."""
        c = self.cfg.cov_channels
        return 0 if not c else n_filters if c == "all" else min(int(c), n_filters)

    def _readout(self, h):
        """h: (windows, filters, time) -> one feature vector per window."""
        c = self.cfg
        if c.readout in ("mean", "meanstd", "meanstd_ppv_max"):  # steps (a) and (b), unchanged
            parts = [h.mean(dim=2)]
            if c.readout in ("meanstd", "meanstd_ppv_max"):
                parts.append(h.std(dim=2, unbiased=True))
            if c.readout == "meanstd_ppv_max":
                parts += [(h > 0).float().mean(dim=2), h.amax(dim=2)]
        elif c.readout == "logvar":   # log band power of each random filter (variance = power around the mean)
            parts = [_log(h.var(dim=2, unbiased=True))]
        elif c.readout == "logpow":   # ShallowConvNet: square -> mean -> log
            parts = [_log(h.pow(2).mean(dim=2))]
        elif c.readout == "subwin":   # log-variance in short sub-windows -> their mean and SD
            n_sub = h.size(2) // c.subwin
            if c.subwin < 2 or n_sub < 2:
                raise ValueError(f"subwin={c.subwin} on {h.size(2)} samples")
            seg = h[:, :, : n_sub * c.subwin].reshape(h.size(0), h.size(1), n_sub, c.subwin)
            lv = _log(seg.var(dim=3, unbiased=True))
            parts = [lv.mean(dim=2), lv.std(dim=2, unbiased=True)]
        elif c.readout == "hjorth":   # activity (log-variance), mobility, complexity of each filter
            d1 = h.diff(dim=2)
            v0, v1, v2 = h.var(dim=2), d1.var(dim=2), d1.diff(dim=2).var(dim=2)
            mob = torch.sqrt(v1 / v0.clamp_min(_TINY))
            parts = [_log(v0), mob, torch.sqrt(v2 / v1.clamp_min(_TINY)) / mob.clamp_min(_TINY)]
        elif c.readout == "none":     # covariance only
            parts = []
        else:
            raise ValueError(c.readout)
        if c.readout_extra == "logvar":
            parts.append(_log(h.var(dim=2, unbiased=True)))
        elif c.readout_extra:
            raise ValueError(c.readout_extra)
        n = self._cov_n(h.size(1))
        if n >= 2:
            z = h[:, :n] - h[:, :n].mean(dim=2, keepdim=True)
            cov = torch.bmm(z, z.transpose(1, 2)) / max(z.size(2) - 1, 1)
            if c.cov_mode == "offdiag":
                iu = torch.triu_indices(n, n, offset=1, device=h.device)
            elif c.cov_mode == "spd":  # full SPD matrix for the tangent space (evaluate.RiemannTangent)
                if c.shrink:
                    tr = torch.diagonal(cov, dim1=1, dim2=2).mean(dim=1)
                    eye = torch.eye(n, dtype=cov.dtype, device=h.device)
                    cov = (1 - c.shrink) * cov + c.shrink * tr[:, None, None] * eye
                iu = torch.triu_indices(n, n, offset=0, device=h.device)  # same order as RiemannTangent
            else:
                raise ValueError(c.cov_mode)
            parts.append(cov[:, iu[0], iu[1]])
        return torch.cat(parts, dim=1)

    def readout_width(self):
        """Filters seen by the readout (single readout block)."""
        conv = self.convs[0] if self.cfg.readout_from == "first_lin" else self.convs[-1]
        return conv.out_channels if hasattr(conv, "out_channels") else conv.pointwise.out_channels

    def spd_blocks(self, feature_dim):
        """[[start column, n]] of the SPD block for evaluate.RiemannTangent, or []."""
        n = self._cov_n(self.readout_width())
        if self.cfg.cov_mode != "spd" or n < 2:
            return []
        if self.cfg.readout_blocks != "last":
            raise ValueError("spd covariance: readout_blocks must be 'last'")
        return [[feature_dim - n * (n + 1) // 2, n]]  # always the last columns

    @torch.no_grad()
    def forward(self, x):
        if self.perm is not None:
            x = x[:, self.perm]
        if self.cfg.readout_from == "first_lin":  # first conv only, before the nonlinearity
            if self.cfg.readout_blocks != "last":
                raise ValueError("readout_from='first_lin' needs readout_blocks='last'")
            return self._readout(self.convs[0](x))
        elif self.cfg.readout_from != "act":
            raise ValueError(self.cfg.readout_from)
        outs = []
        for b, conv in enumerate(self.convs):
            if b > 0:
                x = F.avg_pool1d(x, self.cfg.pool)
            x = self._act(conv(x))
            outs.append(x)
        keep = outs if self.cfg.readout_blocks == "all" else outs[-1:]
        return torch.cat([self._readout(h) for h in keep], dim=1)


# Input versions: number of copies of every electrode in the network input.
N_VERSIONS = {"raw": 1, "unc3": 3, "fb6": 6}
# Filter bank (fb6): the same bands as the handcrafted band powers (handcrafted.BANDS), Hz.
# The phase-1 cache is already band-passed 1-40 Hz (Sleep-EDF 0.5-40), so "gamma" is 30-40 Hz.
FB_BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40)}


def filter_bank(X: np.ndarray, sfreq: float, per_block: int) -> np.ndarray:
    """(n_windows, n_chans, T) -> (n_windows, 6 * n_chans, T): raw, then one band-passed copy per band.

    Channel order is VERSION-major (as unc3): [raw e1..eN, delta e1..eN, theta ..., alpha ...,
    beta ..., gamma e1..eN]. Zero-phase 4th-order Butterworth band-pass (sosfiltfilt), applied to
    each block with its windows joined back in time (as prep.lowpass_blocks), so a 2 s window
    has no filter edge inside a block. Windows after the last full block (never used by
    block_features) are filtered on their own. No labels, nothing fitted.
    """
    from scipy.signal import butter, sosfiltfilt
    n, ch, t = X.shape
    nb = n // per_block
    full = X[: nb * per_block].reshape(nb, per_block, ch, t).transpose(0, 2, 1, 3).reshape(nb, ch, -1)
    out = [X]
    for lo, hi in FB_BANDS.values():
        sos = butter(4, [lo, hi], btype="bandpass", fs=sfreq, output="sos")
        y = sosfiltfilt(sos, full.astype(np.float64), axis=-1)
        y = y.reshape(nb, ch, per_block, t).transpose(0, 2, 1, 3).reshape(nb * per_block, ch, t)
        if n > nb * per_block:  # leftover windows (dropped later by block_features)
            y = np.concatenate([y, sosfiltfilt(sos, X[nb * per_block:].astype(np.float64), axis=-1)])
        out.append(y.astype(np.float32))
    return np.concatenate(out, axis=1)


def prepare_input(X: np.ndarray, cfg: ArchConfig, sfreq: float | None = None,
                  per_block: int | None = None) -> np.ndarray:
    """(n_windows, n_chans, T) float32 -> the network input for this config.
    fb6 needs the sampling rate and the windows per block (one participant's windows, in order)."""
    if cfg.input_versions == "raw":
        return X
    if cfg.input_versions == "unc3":
        from uncnn1d.inputs import build_input
        return np.stack([build_input(w, rank_size=5) for w in X])
    if cfg.input_versions == "fb6":
        if sfreq is None or per_block is None:
            raise ValueError("fb6 input needs sfreq and per_block")
        return filter_bank(X, sfreq, per_block)
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
