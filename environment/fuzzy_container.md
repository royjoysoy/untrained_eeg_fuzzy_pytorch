# Fuzzy PyTorch container

> **PLACEHOLDER — waiting on Roy for the exact image name/tag.**
> Everything marked `TODO(fuzzy-image)` below and in `slurm/mca_variability.sh`
> must be filled in before running goal 3.

[Fuzzy PyTorch](https://github.com/verificarlo/fuzzy) is PyTorch compiled with
[Verificarlo](https://github.com/verificarlo/verificarlo), so that its
floating-point operations are randomly perturbed (Monte Carlo Arithmetic, MCA).
Running the *same* model with the *same* seed several times then shows how much
the result depends on rounding errors.

## 1. Get the image (once)

The images are on Docker Hub as `verificarlo/fuzzy:<version>-pytorch<version>-<cpu>`,
for example `verificarlo/fuzzy:v2.6.0-pytorch2.2.1-avx2`. Rorqual CPUs support
`avx512`. Use one tag for the whole study.

```bash
module load apptainer
# TODO(fuzzy-image): replace with the tag Roy chooses
apptainer pull fuzzy-pytorch.sif docker://verificarlo/fuzzy:TODO-TAG
export FUZZY_IMAGE=$PWD/fuzzy-pytorch.sif
```

## 2. Add the extra Python packages (once)

The image has PyTorch and NumPy but not braindecode, scikit-learn or PyYAML.
Install them into a folder outside the image with `--no-deps`, so pip does not
replace the instrumented PyTorch:

```bash
# TODO(fuzzy-image): check this list against the image; braindecode imports mne.
apptainer exec $FUZZY_IMAGE pip install --no-deps --target=$HOME/fuzzy_pkgs \
    braindecode mne scikit-learn pyyaml pandas
```

The slurm script adds `$HOME/fuzzy_pkgs` to `PYTHONPATH` inside the container.

## 3. Check it works

Outside the container, two runs of `scripts/03_mca_variability.py` give
identical numbers. Inside the container, they should differ slightly:

```bash
apptainer exec --env PYTHONPATH=$HOME/fuzzy_pkgs $FUZZY_IMAGE \
    python3 scripts/03_mca_variability.py --sample 0 --models cnn1d
```

Run it twice and compare the printed AUCs (or look at the probabilities in
`results/mca/`). If they are bit-identical, MCA is not active.

## Notes

- Preprocessing (MNE filtering, resampling) is done **outside** the container
  by script 01 and cached, so only the CNN and probe are perturbed.
- One process = one MCA sample. That is why `03_mca_variability.py` takes `--sample`.
