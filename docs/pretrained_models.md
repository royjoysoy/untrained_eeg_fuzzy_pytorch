# Pretrained scalp-EEG models (inventory)

**Goal:** list every public pretrained checkpoint we can compare against our untrained CNNs,
with what it was pretrained on (to avoid leakage), what input it expects, and its license.
We only use checkpoints as published; we never train or fine-tune. Each model must be copied
into plain PyTorch in `models.py` before it can run in the Fuzzy environment.

Status: first pass, 2026-10-09. "?" means not yet checked: please verify on the linked page
and fill it in. Most weights are on the [braindecode Hugging Face page](https://huggingface.co/braindecode).

## Candidates

Input = channels · sampling rate · window, as in each checkpoint's `config.json` (most models
accept other channel sets through electrode names or positions).

| Model | Checkpoints | Input | Pretraining data | License |
|---|---|---|---|---|
| EEGNetv4 (CNN, supervised) | 12, one per MOABB motor-imagery dataset ([PierreGtch/EEGNetv4](https://huggingface.co/PierreGtch/EEGNetv4)) | 3 (C3, Cz, C4) · 128 Hz · 3 s | One MOABB dataset each: AlexMI, BNCI2014001, BNCI2014004, BNCI2015001, BNCI2015004, Cho2017, Lee2019_MI, Ofner2017, PhysionetMI, Schirrmeister2017, Weibo2014, Zhou2016 | MIT |
| ShallowFBCSPNet (CNN, supervised) | 2, braindecode tutorials ([trial](https://huggingface.co/braindecode/plot_bcic_iv_2a_moabb_trial), [cropped](https://huggingface.co/braindecode/plot_bcic_iv_2a_moabb_cropped)) | 22 (BCI IV 2a) · 250 Hz · 4.5 s | BNCI2014001, subject 3 only, 4 classes | ? (braindecode docs) |
| EEGSym (CNN, supervised) | 1 ([eegsym-original-keras-8ch-pytorch](https://huggingface.co/braindecode/eegsym-original-keras-8ch-pytorch)) | 8 · 128 Hz · 3 s | Motor imagery (?) | other (?) |
| MIRepNet | 1 ([mirepnet-pretrained](https://huggingface.co/braindecode/mirepnet-pretrained)) | 45 · 250 Hz · 4 s | 7 MOABB MI sets: BNCI2014002, PhysionetMI, Dreyer2023, Weibo2014, Zhou2016, Lee2019, Cho2017 ([paper](https://arxiv.org/abs/2507.20254)) | MIT |
| Signal-JEPA | 2 ([signal-jepa](https://huggingface.co/braindecode/signal-jepa), [without channel embeddings](https://huggingface.co/braindecode/signal-jepa_without-chans)); the Contextual/PostLocal/PreLocal repos are deprecated | up to 62 · 128 Hz | Lee2019 (62 channels) | MIT |
| EEGPT | 1 ([eegpt-pretrained](https://huggingface.co/braindecode/eegpt-pretrained)) | 62 · 250 Hz · 4 s | PhysioMI, HGD, TSU, SEED, M3CV ([paper](https://papers.neurips.cc/paper_files/paper/2024/file/4540d267eeec4e5dbd9dae9448f0b739-Paper-Conference.pdf), Table 1) | BSD-3 (?) |
| LaBraM | 1 ([labram-pretrained](https://huggingface.co/braindecode/labram-pretrained)) | up to 128 named channels · 200 Hz (?) | ~20 public sets, 2,535 h: TUH subsets (TUAR, TUEP, TUSZ, TUSL), Siena, Grasp and Lift, BCI Comp IV-1, PhysioNet MMI, SEED family, ... ([paper](https://arxiv.org/abs/2405.18765), App. D) | BSD-3 (?) |
| BENDR | 1 ([braindecode-bendr](https://huggingface.co/braindecode/braindecode-bendr)) | 20 · 250 Hz · 4 s | TUEG | BSD-3 (?) |
| BIOT | 3 ([16 ch](https://huggingface.co/braindecode/biot-pretrained-prest-16chs), [SHHS+PREST 18 ch](https://huggingface.co/braindecode/biot-pretrained-shhs-prest-18chs), [six datasets 18 ch](https://huggingface.co/braindecode/biot-pretrained-six-datasets-18chs)) | 16 or 18 (montage ?) · 200 Hz · 5 s | SHHS (sleep), PREST (private resting EEG); "six datasets" variant ? ([paper](https://arxiv.org/abs/2305.10351)) | BSD-3 (?) |
| CBraMod | 1 ([cbramod-pretrained](https://huggingface.co/braindecode/cbramod-pretrained)) | 19 (10-20) · 200 Hz | TUEG | BSD-3 |
| CodeBrain | 1 ([codebrain-pretrained](https://huggingface.co/braindecode/codebrain-pretrained)) | 19 · 200 Hz · 30 s | TUEG | BSD-3 (?) |
| EEG-DINO | small, medium ([small](https://huggingface.co/braindecode/eegdino-small-pretrained)) | 19 · 200 Hz · 5 s | TUEG (>9,000 h) | BSD-3 |
| ST-EEGFormer | small, base, large, largeV2 ([base](https://huggingface.co/braindecode/STEEGFormer-base)) | any of 145 named channels (256 for largeV2) · ? · 6 s | >8M segments, corpus ?; largeV2 also HBN | MIT |
| BrainOmni | tiny, base ([base](https://huggingface.co/braindecode/brainomni-base-pretrained)) | any (positions) · 256 Hz · 2 s | ~2,000 h public EEG + 656 h MEG, mostly OpenNeuro ([paper](https://arxiv.org/abs/2505.18185), appendix) | MIT |
| DIVER-1 | 2 sizes ([1s-small](https://huggingface.co/braindecode/DIVER-1-1s-small)) | any (positions) · 500 Hz | Intracranial + scalp, ? | MIT |
| NeuroRVQ (EEG) | 1 ([neurorvq-eeg-pretrained](https://huggingface.co/braindecode/neurorvq-eeg-pretrained)) | 19 · 200 Hz · 4 s | ? | CC BY-NC 4.0 |
| SleepFM | 1 ([SleepFM](https://huggingface.co/braindecode/SleepFM)) | 4 · 128 Hz · 30 s | Polysomnography, ? | CC BY-NC 4.0 |
| ZUNA | 1 ([ZUNA](https://huggingface.co/braindecode/ZUNA)) | any (positions) · 256 Hz · 5 s | TUH + 208 public sets, mostly OpenNeuro, list not in the paper ([paper](https://arxiv.org/abs/2602.18478)) | Apache-2.0 |
| REVE | base, large ([brain-bzh/reve-base](https://huggingface.co/brain-bzh/reve-base)) | any (positions) · 200 Hz | 92 sources, 61,415 h: OpenNeuro, MOABB, TUH ([paper](https://arxiv.org/abs/2510.21585), appendix) | Registration required: **excluded** (open weights only) |
| Masking ablations | ~60 MAE/JEPA checkpoints ([PierreGtch](https://huggingface.co/PierreGtch)) | ? | ? | ? |

## Ported

| Model | Loader | Test | Notes |
|---|---|---|---|
| EEGNetv4 (12) | `load_eegnetv4` in `src/untrained_eeg/pretrained.py`; weights from `scripts/00_download_pretrained.sh` | `tests/test_pretrained.py`: bit-identical features to braindecode 1.8.1 | Feed C3, Cz, C4 (`C3, CZ, C4` in our config) at 128 Hz, in microvolts |
| ShallowFBCSPNet (2) | `load_shallow` in `src/untrained_eeg/pretrained.py`; same download script (safetensors) | `tests/test_pretrained.py`: bit-identical features to braindecode 1.8.1 | Feed the 22 BCI IV 2a channels at 250 Hz, microvolts, band-pass 4-38 Hz, exponential moving standardization (factor_new=1e-3, init_block_size=1000) |
| EEGPT, LaBraM, BENDR (interpolated), CBraMod, CodeBrain, EEG-DINO small and medium, Signal-JEPA | `Foundation` in `src/untrained_eeg/foundation.py`, through braindecode 1.8.1 (not plain PyTorch: Fuzzy is paused); weights from the braindecode Hugging Face page, cached in `data/pretrained/hf` | Every pretrained tensor must load (only EEGPT's channel table, rebuilt for our channels, and the heads are left out) | Each gets its pretraining rate (200, 250, 256 or 128 Hz; EEGPT 256 Hz as in the authors' code, not the 250 Hz of braindecode's card), channels (all, the 19 of 10-20, or Lee2019's) and scale; CodeBrain needs 30 s windows. EEGPT in millivolts, as in the authors' pretraining script ([prepare_pretrain_dataset.py](https://github.com/BINE022/EEGPT/blob/ea7282f698c406cd423ccbedbe9960853761144c/datasets/pretrain/prepare_pretrain_dataset.py)); BENDR min-max scaled is an assumption |

BENDR as a frozen extractor: we could not reproduce its pretraining input. With our input its
features are nearly constant (median SD across blocks 0.0004, against 0.2 to 0.9 for EEGPT and
LaBraM), so every task is near chance. The likely cause is on our side: its input scaling is not
documented (we assume per-channel min-max), or braindecode's InterpolatedBENDR returns other
features than the authors' code; this is not evidence about BENDR itself. It is therefore left
out of the ensemble pools and of every test family (`EXCLUDED` in `src/untrained_eeg/foundation.py`); its AUCs are still computed and
reported, and it comes back in if a fix gives features with a normal spread.

Input scale of EEGNetv4: the cards do not say. The first batch-norm's running variance gives
the input SD seen in training: 3–35 for 10 checkpoints (microvolts, like our cache), but ~340
for Cho2017 and ~8×10⁶ for Ofner2017 (another unit). Rescale for those two, or leave them out.

## Leakage against our candidate tasks

A pretrained model can only be evaluated on a dataset it was not pretrained on. Checked by
searching each paper for the dataset names and OpenNeuro IDs; "ok" = not found in the
pretraining list, "?" = list not checked or not published.

| Model | ds003478 (EC/EO, BDI) | ds004504 (AD/FTD) | ds002778 (PD) | LEMON (age) | BNCI2014001 (MI) | Sleep-EDF |
|---|---|---|---|---|---|---|
| EEGNetv4 | ok | ok | ok | ok | **leak** for the BNCI2014001 checkpoint: use the other 11 | ok |
| ShallowFBCSPNet | ok | ok | ok | ok | **leak**: both checkpoints | ok |
| MIRepNet | ok | ok | ok | ok | ok (it is MIRepNet's downstream set) | ok |
| Signal-JEPA | ok | ok | ok | ok | ok | ok |
| EEGPT | ok | ok | ok | ok | ok (downstream set) | ok (downstream set) |
| LaBraM | ok | ok | ok | ok | ok | ok |
| BENDR, CBraMod, CodeBrain, EEG-DINO (TUEG only) | ok | ok | ok | ok | ok | ok |
| BrainOmni | ok | ok (downstream set) | ok (downstream set) | ? | ? | ? |
| REVE | ? | ? | **leak** (ds002778 in pretraining) | ? | ? | ? |
| ZUNA | ? | ? | ? | ? | ? | ? |
| ST-EEGFormer | ? | ? | ? | ? | ? | ? |
| BIOT, DIVER-1, NeuroRVQ, SleepFM | ? | ? | ? | ? | ? | ? |

## Out of scope

- **Intracranial (sEEG/iEEG):** BrainBERT, PopT, Brant, MVPFormer, BaRISTA, MAPA.
- **Tokenizers, not encoders:** BrainTokenizer, NeuroRVQ tokenizers.
- **Other signals:** EMG, ECG and PPG models on the same Hugging Face page.
- **No public weights:** architectures listed on braindecode without a `-pretrained` repository.

## Decisions (Roy, 2026-10-09)

- Open weights only: REVE (registration required) is excluded.
- CC BY-NC checkpoints (NeuroRVQ, SleepFM) are fine.
- ZUNA pretrains on large parts of OpenNeuro: keep it only for tasks we can prove are
  outside its corpus.

## Related work

- [Are foundation models useful feature extractors for EEG analysis?](https://arxiv.org/abs/2502.21086)
  (frozen features, close to our question).
- [Are EEG foundation models worth it?](https://openreview.net/forum?id=5Xwm8e6vbh) (ICLR 2026,
  ST-EEGFormer; foundation models vs compact decoders).
