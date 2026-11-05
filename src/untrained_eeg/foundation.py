"""Pretrained EEG foundation models as frozen feature extractors (braindecode 1.8.1).

Unlike the CNNs in models.py, these run through braindecode itself: the Fuzzy PyTorch
environment is paused, so they are not yet copied into plain PyTorch. Weights come from
the braindecode Hugging Face page (bash scripts/00_download_pretrained.sh), read offline.

Each model gets the input it was pretrained on: its sampling rate (a preprocessing variant,
see `VARIANT`), its channels (all the dataset's, or the 19 of the 10-20 system), its window
length and its amplitude scale. Its features, averaged over time patches with channel and
embedding axes kept, play the role of the CNN features. How to average is set per model
(`TOKENS`), not guessed from the output's shape.

  eegpt          EEGPT, 256 Hz, the channels in its table, millivolts (as in the authors' pretraining
                 script; braindecode's model card says 250 Hz, the authors pretrain and evaluate at 256)
  labram         LaBraM, 200 Hz, all channels, microvolts / 100 (0.1 mV units)
  bendr          InterpolatedBENDR, 250 Hz, any montage projected onto BENDR's 19 + SCALE,
                 each channel min-max scaled to [-1, 1] (assumed from BENDR's pipeline)
  cbramod        CBraMod, 200 Hz, all channels, microvolts / 100
  codebrain      CodeBrain, 200 Hz, 19 channels (10-20), 30 s windows, microvolts / 100
  eegdino_small  EEG-DINO small, 200 Hz, 19 channels (10-20), microvolts / 100
  eegdino_medium EEG-DINO medium, same input
  signal_jepa    Signal-JEPA with its Lee2019 channel embeddings, 128 Hz, the dataset's
                 channels that Lee2019 has, microvolts
  reve_base      REVE base (authors' brain-bzh repository), 200 Hz, all channels placed by
                 REVE's position bank, each channel z-scored over the participant's recording
                 and clipped at 15 SD, as in its pretraining (`zscore_recording`)
  reve_large     REVE large, same input
  neurogpt       Neuro-GPT's EEG encoder (an EEG Conformer, authors' wenhuic repository),
                 250 Hz, its 22 TUH channels (channels the dataset lacks interpolated by
                 `tuh22_matrix`) re-referenced to their average, each channel z-scored over the
                 participant's recording, 2 s chunks; features = the 1,080-value chunk embedding
                 its GPT reads (27 time tokens x 40)
"""

import importlib
import inspect
import json
import os

import numpy as np
import torch
from torch import nn

from .data import REPO

HF_HOME = REPO / "data" / "pretrained" / "hf"  # set as HF_HOME by the scripts and jobs
REVE_POSITIONS = REPO / "data" / "pretrained" / "reve"  # REVE's position bank (reve_positions.json)

# Neuro-GPT's 22 channels, in its order (T3-T6 under their new names; T1, T2 placed at FT9, FT10,
# which they approximately are)
TUH22 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T1", "T7", "C3", "Cz", "C4", "T8", "T2",
         "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]
AT = {"T1": "FT9", "T2": "FT10"}
TEN20 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8",
         "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}  # old 10-20 names

FOUNDATION = {  # key: class, repo, sfreq, channels, window_s (None = the dataset's), scale, how
    "eegpt": ("EEGPT", "braindecode/eegpt-pretrained", 256, "all", None, "mv", "manual"),
    "labram": ("Labram", "braindecode/labram-pretrained", 200, "all", None, "0.1mv", "manual"),
    "bendr": ("InterpolatedBENDR", "braindecode/braindecode-bendr", 250, "all", None, "minmax", "manual"),
    "cbramod": ("CBraMod", "braindecode/cbramod-pretrained", 200, "all", None, "0.1mv", "hub"),
    "codebrain": ("CodeBrain", "braindecode/codebrain-pretrained", 200, "ten20", 30, "0.1mv", "hub"),
    "eegdino_small": ("EEGDINO", "braindecode/eegdino-small-pretrained", 200, "ten20", None, "0.1mv", "hub"),
    "eegdino_medium": ("EEGDINO", "braindecode/eegdino-medium-pretrained", 200, "ten20", None, "0.1mv", "hub"),
    "signal_jepa": ("SignalJEPA", "braindecode/signal-jepa", 128, "lee2019", None, "uv", "hub"),
    "reve_base": ("REVE", "brain-bzh/reve-base", 200, "all", None, "zscore", "manual"),
    "reve_large": ("REVE", "brain-bzh/reve-large", 200, "all", None, "zscore", "manual"),
    "neurogpt": ("EEGConformer", "wenhuic/Neuro-GPT", 250, "tuh22", None, "zscore_noclip", "neurogpt"),
}
VARIANT = {200: "fm200", 250: "fm250", 256: "fm256", 128: "fm128"}  # sampling rate -> preprocessing variant
# Extracted but left out of the ensemble pools and of every test family, for a reason found on the
# features themselves (not on the results); their AUCs are still reported (docs/pretrained_models.md)
# BENDR: we could not reproduce its pretraining input (undocumented scaling), and its features
# were nearly constant; this says nothing about BENDR itself.
EXCLUDED = {"bendr": "input not reproduced: near-constant features (median SD across blocks 0.0004, "
                     "0.2-0.9 for EEGPT and LaBraM)"}
# Models whose features are (batch, tokens, dim): "channels" = channel-major tokens
# (channels * patches), averaged over each channel's patches; "time" = time tokens, averaged.
# Others return (batch, channels, patches, dim), averaged over patches, or (batch, dim).
TOKENS = {"labram": "channels", "eegdino_small": "channels", "eegdino_medium": "channels",
          "signal_jepa": "channels", "eegpt": "time", "neurogpt": "flat"}
NEUROGPT_TIMES = 500  # Neuro-GPT's encoder reads 2 s chunks at 250 Hz


def zscore_recording(X, clip=15.0):
    """REVE's input: each channel z-scored over one participant's windows (n_windows, n_chans,
    n_times), then clipped at 15 SD, as REVE's pretraining did per recording (clip=None: not
    clipped, as for Neuro-GPT)."""
    mean, sd = X.mean(axis=(0, 2), keepdims=True), X.std(axis=(0, 2), keepdims=True)
    Z = (X - mean) / np.maximum(sd, 1e-6)
    return (Z if clip is None else np.clip(Z, -clip, clip)).astype(X.dtype)


def tuh22_matrix(names):
    """(22, n) matrix taking the dataset's n channels to Neuro-GPT's 22, average-referenced.

    Channels the dataset has are copied; the others are interpolated with spherical splines
    from all its channels (MNE's interpolate_bads, run on an identity matrix to get the
    weights), then all 22 are re-referenced to their average, as in Neuro-GPT's pretraining.
    Its authors mapped their own downstream data (BCI IV 2a) the same way, with another
    interpolation (a forward and inverse model).
    """
    import mne

    placed = set(mne.channels.make_standard_montage("standard_1005").ch_names)
    used = [c for c in names if c in placed]  # channels without a position (e.g. EOG) are not used
    targets = [AT.get(c, c) for c in TUH22]
    missing = [c for c in targets if c not in used]
    info = mne.create_info(used + missing, 250.0, "eeg")
    info.set_montage("standard_1005", verbose="error")
    info["bads"] = missing
    eye = np.vstack([np.eye(len(used)), np.zeros((len(missing), len(used)))])
    W = mne.EvokedArray(eye, info, verbose="error").interpolate_bads(origin="auto", verbose="error").data
    out = np.zeros((len(targets), len(names)))
    out[:, [names.index(c) for c in used]] = W[[info.ch_names.index(c) for c in targets]]
    return (out - out.mean(axis=0)).astype(np.float32)


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

    if cls_name == "REVE":  # its position bank, prefetched by scripts/00_download_pretrained.sh
        os.environ.setdefault("REVE_POSITIONS_PATH", str(REVE_POSITIONS))
    kw = dict(n_outputs=2, n_chans=len(names), n_times=n_times, chs_info=_chs_info(names, sfreq), sfreq=sfreq)
    cls = getattr(M, cls_name)
    if how == "hub":
        model = cls.from_pretrained(repo, **kw)
    elif how == "neurogpt":  # an EEG Conformer of 6 layers; the original's other settings are braindecode's defaults
        model = cls(**kw, num_layers=6, return_features=True)
        load_neurogpt(model, hf_hub_download(repo, "pretrained_model/pytorch_model.bin"))
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


def load_neurogpt(model, path):
    """Neuro-GPT's encoder weights (its GPT is left out) into braindecode's EEGConformer.

    The names match once the "encoder." prefix is removed. The original attention divides by
    sqrt(40) (the embedding size), braindecode's by sqrt(4) (the head size): the query
    weights are scaled by sqrt(4 / 40) so the attention is the original one.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = {k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith(("fc.", "final_layer.")) for k in missing):
        raise RuntimeError(f"Neuro-GPT weights do not fit: missing {missing}, unexpected {unexpected}")
    with torch.no_grad():
        for block in model.transformer:
            att = block[0].fn[1]
            att.queries.weight.mul_((att.head_dim / att.emb_size) ** 0.5)
            att.queries.bias.mul_((att.head_dim / att.emb_size) ** 0.5)


def load_weights(model, path):
    """Load a checkpoint into a model built for another input; return what was left out.

    - LaBraM: its time embedding has one row per 1 s patch of its 15 s pretraining window;
      the first rows are used for shorter windows, as the original code does.
    - EEGPT: `chans_id` is a table of channel indices, not weights; ours is kept.
    - REVE: `cls_query_token` serves attention pooling only, which we do not use.
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
    """A foundation model fed the dataset's windows (batch, n_chans, n_times) in microvolts
    (REVE and Neuro-GPT: after `prepare`, over the participant's windows)."""

    def __init__(self, key, ch_names, n_times):
        super().__init__()
        cls_name, repo, sfreq, chans, _, self.scale, how = FOUNDATION[key]
        names = standard_names(ch_names)
        self.mix = None  # Neuro-GPT: (22, n_chans) channel mapping, applied in prepare()
        if chans == "ten20":
            want = TEN20
        elif chans == "tuh22":  # mixed in prepare(): the model sees the 22 channels in order
            self.mix = tuh22_matrix(names)
            names = want = TUH22
        elif chans == "lee2019":
            lee = {c.upper() for c in _lee2019_names()}
            want = [c for c in names if c.upper() in lee]
        else:
            want = names
        if cls_name == "EEGPT":  # channels outside its table would get another channel's embedding
            from braindecode.models.eegpt import CHANNEL_DICT

            want = [c for c in want if c.upper() in CHANNEL_DICT]
        missing = [c for c in want if c not in names]
        if missing:
            raise ValueError(f"{key}: the dataset lacks {missing}")
        self.register_buffer("idx", torch.tensor([names.index(c) for c in want]), persistent=False)
        self.n_chans = len(want)
        self.tokens = TOKENS.get(key)
        self.model = _build(cls_name, repo, want, sfreq, n_times, how)
        if cls_name == "REVE" and len(self.model.default_pos) != self.n_chans:  # it only warns
            raise ValueError(f"{key}: REVE's position bank lacks some of {want}")
        if cls_name == "EEGConformer" and n_times != NEUROGPT_TIMES:
            raise ValueError(f"{key}: takes 2 s chunks at 250 Hz ({NEUROGPT_TIMES} samples), not {n_times}")
        self.fwd = {"ch_names": want} if cls_name == "Labram" else {}
        if cls_name != "EEGConformer":  # built with return_features=True
            self.fwd["return_features"] = True

    def prepare(self, X):
        """Steps over one participant's windows (its recording, numpy), before the model:
        Neuro-GPT's channel mapping, and REVE's and Neuro-GPT's z-score per recording."""
        if self.mix is not None:
            X = np.einsum("ij,wjt->wit", self.mix, X)
        if self.scale == "zscore":
            X = zscore_recording(X)
        elif self.scale == "zscore_noclip":
            X = zscore_recording(X, clip=None)
        return X

    def forward(self, x):
        x = x[:, self.idx]
        if self.scale == "0.1mv":
            x = x / 100
        elif self.scale == "mv":
            x = x / 1000
        elif self.scale == "minmax":  # per window and channel, to [-1, 1]
            lo, hi = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
            x = 2 * (x - lo) / (hi - lo).clamp_min(1e-6) - 1
        out = self.model(x, **self.fwd)
        f = out["features"] if isinstance(out, dict) else out
        if f.dim() == 4:  # (batch, channels, patches, dim): average the time patches
            f = f.mean(dim=2)
        elif f.dim() == 3 and self.tokens == "channels":  # (batch, channels * patches, dim)
            if f.shape[1] % self.n_chans:
                raise ValueError(f"{f.shape[1]} tokens for {self.n_chans} channels")
            f = f.reshape(f.shape[0], self.n_chans, -1, f.shape[-1]).mean(dim=2)
        elif f.dim() == 3 and self.tokens == "time":  # (batch, time tokens, dim)
            f = f.mean(dim=1)
        elif f.dim() == 3 and self.tokens == "flat":  # Neuro-GPT: the chunk embedding, as its GPT reads it
            f = f.flatten(1)
        elif f.dim() != 2:
            raise ValueError(f"unexpected feature shape {tuple(f.shape)}: set the model's TOKENS")
        return f
