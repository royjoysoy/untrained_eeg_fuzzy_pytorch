"""Pretrained EEG foundation models as frozen feature extractors (braindecode 1.8.1).

Unlike the CNNs in models.py, these run through braindecode itself: the Fuzzy PyTorch
environment is paused, so they are not yet copied into plain PyTorch. Weights come from
the braindecode Hugging Face page (bash scripts/00_download_pretrained.sh), read offline.

Each model gets the input it was pretrained on: its sampling rate (a preprocessing variant,
see `VARIANT`), its channels (all the dataset's, or the 19 of the 10-20 system), its window
length and its amplitude scale. Its features, averaged over time patches with channel and
embedding axes kept, play the role of the CNN features. How to average is set per model
(`TOKENS`), not guessed from the output's shape.

  eegpt          EEGPT, 250 Hz, all channels, millivolts (as in the authors' pretraining script)
  labram         LaBraM, 200 Hz, all channels, microvolts / 100 (0.1 mV units)
  bendr          InterpolatedBENDR, 250 Hz, any montage projected onto BENDR's 19 + SCALE,
                 each channel min-max scaled to [-1, 1] (assumed from BENDR's pipeline)
  cbramod        CBraMod, 200 Hz, all channels, microvolts / 100
  codebrain      CodeBrain, 200 Hz, 19 channels (10-20), 30 s windows, microvolts / 100
  eegdino_small  EEG-DINO small, 200 Hz, 19 channels (10-20), microvolts / 100
  eegdino_medium EEG-DINO medium, same input
  signal_jepa    Signal-JEPA with its Lee2019 channel embeddings, 128 Hz, the dataset's
                 channels that Lee2019 has, microvolts
"""

import importlib
import inspect
import json

import numpy as np
import torch
from torch import nn

from .data import REPO

HF_HOME = REPO / "data" / "pretrained" / "hf"  # set as HF_HOME by the scripts and jobs

TEN20 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8",
         "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}  # old 10-20 names

FOUNDATION = {  # key: class, repo, sfreq, channels, window_s (None = the dataset's), scale, how
    "eegpt": ("EEGPT", "braindecode/eegpt-pretrained", 250, "all", None, "mv", "manual"),
    "labram": ("Labram", "braindecode/labram-pretrained", 200, "all", None, "0.1mv", "manual"),
    "bendr": ("InterpolatedBENDR", "braindecode/braindecode-bendr", 250, "all", None, "minmax", "manual"),
    "cbramod": ("CBraMod", "braindecode/cbramod-pretrained", 200, "all", None, "0.1mv", "hub"),
    "codebrain": ("CodeBrain", "braindecode/codebrain-pretrained", 200, "ten20", 30, "0.1mv", "hub"),
    "eegdino_small": ("EEGDINO", "braindecode/eegdino-small-pretrained", 200, "ten20", None, "0.1mv", "hub"),
    "eegdino_medium": ("EEGDINO", "braindecode/eegdino-medium-pretrained", 200, "ten20", None, "0.1mv", "hub"),
    "signal_jepa": ("SignalJEPA", "braindecode/signal-jepa", 128, "lee2019", None, "uv", "hub"),
}
VARIANT = {200: "fm200", 250: "fm250", 128: "fm128"}  # sampling rate -> preprocessing variant
# Models whose features are (batch, tokens, dim): "channels" = channel-major tokens
# (channels * patches), averaged over each channel's patches; "time" = time tokens, averaged.
# Others return (batch, channels, patches, dim), averaged over patches, or (batch, dim).
TOKENS = {"labram": "channels", "eegdino_small": "channels", "eegdino_medium": "channels",
          "signal_jepa": "channels", "eegpt": "time"}


def variant_of(key):
    _, _, sfreq, _, window_s, _, _ = FOUNDATION[key]
    return "fm200_30s" if window_s == 30 else VARIANT[sfreq]


def standard_names(names):
    """Channel names in MNE's standard_1005 spelling (Fp1, Cz), old 10-20 names updated."""
    import mne

    spell = {c.upper(): c for c in mne.channels.make_standard_montage("standard_1005").ch_names}
    return [spell.get(ALIASES.get(c, c).upper(), c) for c in names]


def _chs_info(names, sfreq):
    import mne

    info = mne.create_info(names, sfreq, "eeg")
    info.set_montage("standard_1005", on_missing="ignore", verbose="error")
    return info["chs"]


def _lee2019_names():
    from huggingface_hub import hf_hub_download

    cfg = json.load(open(hf_hub_download("braindecode/signal-jepa", "config.json")))
    return [c["ch_name"] for c in cfg["chs_info"]]


def _build(cls_name, repo, names, sfreq, n_times, how):
    """The braindecode model with its pretrained weights, in eval mode."""
    import braindecode.models as M
    from huggingface_hub import hf_hub_download

    kw = dict(n_outputs=2, n_chans=len(names), n_times=n_times, chs_info=_chs_info(names, sfreq), sfreq=sfreq)
    cls = getattr(M, cls_name)
    if how == "hub":
        model = cls.from_pretrained(repo, **kw)
    else:  # build for our input, then load what fits (see the notes in load_weights)
        cfg = json.load(open(hf_hub_download(repo, "config.json")))
        sig = inspect.signature(cls.__init__).parameters
        open_kw = any(p.kind == p.VAR_KEYWORD for p in sig.values())
        skip = {"n_outputs", "n_chans", "chs_info", "n_times", "sfreq", "input_window_seconds",
                "n_chans_pretrained", "braindecode_version", "chan_proj_max_norm"}
        extra = {k: v for k, v in cfg.items() if k not in skip and (open_kw or k in sig)}
        for k, v in list(extra.items()):  # "torch.nn.modules.activation.GELU" -> the class
            if isinstance(v, str) and v.startswith("torch."):
                mod, _, attr = v.rpartition(".")
                extra[k] = getattr(importlib.import_module(mod), attr)
        if cls_name == "EEGPT":
            extra["chan_proj_type"] = "none"  # its channel table is built for our channels
        model = cls(**kw, **extra)
        load_weights(model, hf_hub_download(repo, "model.safetensors"))
    return model.eval()


def load_weights(model, path):
    """Load a checkpoint into a model built for another input; return what was left out.

    - LaBraM: its time embedding has one row per 1 s patch of its 15 s pretraining window;
      the first rows are used for shorter windows, as the original code does.
    - EEGPT: `chans_id` is a table of channel indices, not weights; ours is kept.
    - The classification head is never loaded (we use features only).
    """
    from safetensors.torch import load_file

    state, own = load_file(path), model.state_dict()
    if "temporal_embedding" in state and "temporal_embedding" in own:
        state["temporal_embedding"] = state["temporal_embedding"][:, : own["temporal_embedding"].shape[1]]
    keep = {k: v for k, v in state.items() if k in own and v.shape == own[k].shape}
    model.load_state_dict(keep, strict=False)
    left = [k for k in own if k not in keep and not k.startswith("final_layer")]
    if any(k != "chans_id" for k in left):
        raise RuntimeError(f"pretrained weights missing for {left}")
    return left


class Foundation(nn.Module):
    """A foundation model fed the dataset's windows (batch, n_chans, n_times) in microvolts."""

    def __init__(self, key, ch_names, n_times):
        super().__init__()
        cls_name, repo, sfreq, chans, _, self.scale, how = FOUNDATION[key]
        names = standard_names(ch_names)
        if chans == "ten20":
            want = TEN20
        elif chans == "lee2019":
            lee = {c.upper() for c in _lee2019_names()}
            want = [c for c in names if c.upper() in lee]
        else:
            want = names
        missing = [c for c in want if c not in names]
        if missing:
            raise ValueError(f"{key}: the dataset lacks {missing}")
        self.register_buffer("idx", torch.tensor([names.index(c) for c in want]), persistent=False)
        self.n_chans = len(want)
        self.tokens = TOKENS.get(key)
        self.model = _build(cls_name, repo, want, sfreq, n_times, how)
        self.fwd = {"ch_names": want} if cls_name == "Labram" else {}

    def forward(self, x):
        x = x[:, self.idx]
        if self.scale == "0.1mv":
            x = x / 100
        elif self.scale == "mv":
            x = x / 1000
        elif self.scale == "minmax":  # per window and channel, to [-1, 1]
            lo, hi = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
            x = 2 * (x - lo) / (hi - lo).clamp_min(1e-6) - 1
        out = self.model(x, return_features=True, **self.fwd)
        f = out["features"] if isinstance(out, dict) else out
        if f.dim() == 4:  # (batch, channels, patches, dim): average the time patches
            f = f.mean(dim=2)
        elif f.dim() == 3 and self.tokens == "channels":  # (batch, channels * patches, dim)
            if f.shape[1] % self.n_chans:
                raise ValueError(f"{f.shape[1]} tokens for {self.n_chans} channels")
            f = f.reshape(f.shape[0], self.n_chans, -1, f.shape[-1]).mean(dim=2)
        elif f.dim() == 3 and self.tokens == "time":  # (batch, time tokens, dim)
            f = f.mean(dim=1)
        elif f.dim() != 2:
            raise ValueError(f"unexpected feature shape {tuple(f.shape)}: set the model's TOKENS")
        return f
