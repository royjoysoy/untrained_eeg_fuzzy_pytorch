# Running on Rorqual, step by step

Rorqual is a Digital Research Alliance of Canada cluster. You need an Alliance
account and membership in an allocation (ask Roy which one to use).

## 1. Log in and get the code

```bash
ssh <username>@rorqual.alliancecan.ca
cd ~/projects/<allocation>/<username>       # or anywhere in your space
git clone https://github.com/royjoysoy/untrained_eeg_fuzzy_pytorch.git
cd untrained_eeg_fuzzy_pytorch
```

## 2. Make a Python environment (once)

```bash
module load python/3.11
python -m venv ~/venv/untrained_eeg
source ~/venv/untrained_eeg/bin/activate
pip install --upgrade pip
pip install -r environment/requirements.txt
```

The slurm scripts expect this venv at `~/venv/untrained_eeg`. To use another
location, `export VENV=/your/venv` before `sbatch`.

## 3. Point to the data

Please don't download your own copy if a shared one exists.

```bash
# TODO(Roy): shared copy in the group /project space, e.g.
export UNTRAINED_EEG_DATA=/project/<allocation>/shared/ds003478
```

Add that line to your `~/.bashrc` so you don't have to repeat it.
If there is no shared copy yet: `bash scripts/00_download_data.sh <folder>`.

## 4. Run the steps as jobs

Login nodes are shared, so run anything longer than a minute or two as a job.
Run these from the repo root. In each script, change `#SBATCH --account` to your allocation.

```bash
sbatch slurm/prepare_cache.sh       # goal 1: preprocess + cache EEG, one model run (~10 min)
sbatch slurm/seed_variability.sh    # goal 2: all seeds (~15 min)
sbatch slurm/mca_variability.sh     # goal 3: Fuzzy PyTorch array job (needs FUZZY_IMAGE, see below)
```

Wait for `prepare_cache.sh` to finish before starting the other two
(or use `sbatch --dependency=afterok:<jobid> ...`).

For goal 3, first set up the container: [environment/fuzzy_container.md](../environment/fuzzy_container.md).

## 5. Check on jobs

```bash
sq                                  # your queued and running jobs
tail -f slurm/logs/eeg-seeds-*.out  # follow a log
```

## 6. Compare (goal 4)

This step is quick, so it can run on the login node:

```bash
source ~/venv/untrained_eeg/bin/activate
python scripts/04_compare.py
```

Figures land in `results/figures/`, numbers in `results/summary.csv`.
