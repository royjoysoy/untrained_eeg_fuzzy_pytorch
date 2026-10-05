# Contributing (Brainhack onboarding)

Welcome! All levels are welcome. No EEG or deep-learning background needed for the beginner goals.

## 1. Pick a goal

| Goal | Level | Start here |
|---|---|---|
| List open EEG datasets | Beginner | `docs/datasets.md` |
| Run an untrained CNN on one dataset | Beginner | `notebooks/quickstart.ipynb` |
| Compare across seeds | Intermediate | `scripts/02_seed_variability.py` |
| Run with Fuzzy PyTorch (MCA) | Intermediate | `environment/fuzzy_container.md` |
| Seed vs MCA variability | Advanced | `scripts/04_compare.py` |
| Short paper draft | Final | `paper/` |

Tell the team which goal you picked, in person or on the matching GitHub issue.

## 2. Set up

Follow the Quickstart in the [README](README.md). On Rorqual, use [docs/rorqual.md](docs/rorqual.md).

## 3. Branch, commit, open a pull request

```bash
git checkout -b yourname/short-description   # e.g. mina/eegnet-dropout
# ...edit...
git add <files>
git commit -m "Short description of the change"
git push -u origin yourname/short-description
```

Then open a pull request on GitHub. Small PRs are best. Someone from the team will review it.
No write access? Fork the repo first and open the PR from your fork.

## House rules

- **Never commit data or results.** `data/` and `results/` are git-ignored; keep it that way.
- **No hardcoded paths.** Use the config or the `UNTRAINED_EEG_DATA` environment variable.
- Keep files short and readable. Plain names, a comment wherever a beginner might wonder "why?".
- Change a setting in `configs/ds003478.yaml` rather than in the code.
- Not sure about a scientific choice? Ask Roy (@royjoysoy) instead of guessing.
