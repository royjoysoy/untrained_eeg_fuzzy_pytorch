# How stable are untrained CNNs?

**Measuring random-weight CNN variability on EEG with Fuzzy PyTorch**
A Brainhack Montreal (Fall 2026) project. Status: prototype.

Untrained CNNs, whose weights are random and never trained, are often used as
baselines or as cheap feature extractors. How much do their results depend on
chance? We look at two sources of chance:

1. **Seed variability:** a different random initialization of the weights.
2. **Numerical variability:** floating-point rounding. We measure it with
   [Fuzzy PyTorch](https://github.com/verificarlo/fuzzy), which runs the *same*
   model many times with randomly perturbed arithmetic (Monte Carlo Arithmetic, MCA).

Then we compare the two.

**Setup.** Resting EEG from [OpenNeuro ds003478](https://openneuro.org/datasets/ds003478/versions/1.1.0)
(122 participants, 64 channels). We take run-01 and cut it into 1-minute eyes-open
and eyes-closed blocks, then 2-second windows. Three random-weight CNNs (a small
1D CNN, ShallowConvNet and EEGNet, the last two copied from braindecode) turn each window into features.
A logistic-regression probe classifies **high vs low depression score (BDI)** from
each participant's average features. Variability is measured on the probe's outputs.

```
EEG windows ──► untrained CNN (seed s, or MCA sample m) ──► features ──► probe ──► P(high BDI), AUC
```

## Quickstart

```bash
git clone https://github.com/royjoysoy/untrained_eeg_fuzzy_pytorch.git
cd untrained_eeg_fuzzy_pytorch
python -m venv ~/venv/untrained_eeg && source ~/venv/untrained_eeg/bin/activate
pip install -r environment/requirements.txt

bash scripts/00_download_data.sh /path/to/ds003478   # ~10 GB, skip if you have it
export UNTRAINED_EEG_DATA=/path/to/ds003478

python scripts/01_run_untrained.py   # first run preprocesses + caches the EEG (~5 min)
```

You should see something like:

```
113 participants: 74 low BDI, 39 high BDI
cnn1d, eyes closed, seed 0: AUC = 0.654, balanced accuracy = 0.578
```

On the Rorqual cluster, follow [docs/rorqual.md](docs/rorqual.md) instead.
Prefer a notebook? Open [notebooks/quickstart.ipynb](notebooks/quickstart.ipynb).

## Brainhack goals → where to work

| Goal | Level | Where |
|---|---|---|
| List open EEG datasets | Beginner | [docs/datasets.md](docs/datasets.md) |
| Run an untrained CNN on one dataset | Beginner | `scripts/01_run_untrained.py`, `notebooks/quickstart.ipynb` |
| Compare across seeds | Intermediate | `scripts/02_seed_variability.py` |
| Run with Fuzzy PyTorch (MCA) | Intermediate | `scripts/03_mca_variability.py`, [environment/fuzzy_container.md](environment/fuzzy_container.md) |
| Seed vs MCA variability | Advanced | `scripts/04_compare.py` |
| Short paper draft | Final | [paper/](paper/) |

New here? Read [CONTRIBUTING.md](CONTRIBUTING.md).

## Repository layout

```
configs/ds003478.yaml   all settings (data path, preprocessing, models, seeds, BDI cutoffs)
src/untrained_eeg/
  data.py               load EEG, cut eyes-open/closed blocks, cache windows
  models.py             the three random-weight CNNs
  variability.py        features -> probe -> variability metrics
tests/                  pytest checks (python -m pytest tests/)
scripts/                00-04, one per goal, run in order
slurm/                  Rorqual jobs (cache, seeds, MCA array)
environment/            requirements.txt + Fuzzy PyTorch container notes
docs/                   datasets list, Rorqual guide
notebooks/              quickstart
paper/                  short paper draft
data/, results/         created when you run things; never committed
```

## What gets measured

For each model and condition (eyes open, eyes closed), across runs:

- **Probe AUC**: mean, standard deviation, range.
- **Per-participant P(high BDI)**: standard deviation across runs, and the number
  of **significant digits** shared by all runs, `-log10(std / |mean|)`
  (the usual MCA estimate; 15.9 means the runs agree to the last float64 digit).

`scripts/04_compare.py` writes `results/summary.csv` and two figures.

## Data notes

- Participant with original ID **544** is excluded (unstable BDI, as in the original study).
- Labels: BDI ≤ 6 = low, BDI ≥ 17 = high, others are dropped (original study: < 7 vs > 16).
- Only run-01 is used. Blocks shorter than 1 minute are skipped, so a few
  participants have fewer than six blocks.
- 60 scalp channels are kept; mastoids, CB1/CB2, EOG and EKG are dropped.

## Contributors

| Name | Role | GitHub |
|---|---|---|
| Roy Seo | Lead, postdoc, CAMH / KCNI | [@royjoysoy](https://github.com/royjoysoy) |
| Yohan Chatelain | Scientific associate, CAMH / KCNI | [@yohanchatelain](https://github.com/yohanchatelain) |
| Mina Alizadeh | PhD student | [@mina94az](https://github.com/mina94az) |
| Injy Fouda | Master's student | [@injyf](https://github.com/injyf) |
| Jishang Jiang | Contributor | [@GingerMonke](https://github.com/GingerMonke) |

## References

- Gonzalez-Pepe, Akhaddar, Glatard & Chatelain (2026). *Fuzzy PyTorch: Rapid Numerical Variability Evaluation for Deep Learning Models.* TMLR. https://openreview.net/forum?id=0ogq232VGP
- Denis, de Oliveira Castro & Petit (2016). *Verificarlo: Checking Floating Point Accuracy through Monte Carlo Arithmetic.* IEEE ARITH.
- Cavanagh (2021). *EEG: Depression rest.* OpenNeuro ds003478 v1.1.0. doi:10.18112/openneuro.ds003478.v1.1.0

See [CITATION.cff](CITATION.cff) to cite this repository.

## License

[MIT](LICENSE). The dataset is CC0 and is not included in this repository.
