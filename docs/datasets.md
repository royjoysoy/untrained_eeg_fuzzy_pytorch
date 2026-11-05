# Datasets

Six open EEG datasets, eight tasks. None is in this repository: each is downloaded into
`data/raw/<dataset>` the first time a script needs it (or read from an existing copy, see
`data_dir` / `data_env` in its config). Every number below comes from the config, the
dataset's own files, or the outputs of `scripts/phase1_probe.py`.

| Dataset | Task(s) | Participants used | Probe sample | Channels, rate (raw) | License |
|---|---|---|---|---|---|
| [ds003478](#ds003478-depression-rest) | Eyes closed vs open; high vs low BDI | 113 (74 low, 39 high BDI) | 1-min block; participant | 64, 500 Hz | CC0 |
| [ds004504](#ds004504-alzheimers-and-frontotemporal-dementia) | AD vs control; FTD vs control | 88 (36 AD, 23 FTD, 29 controls) | participant | 19, 500 Hz | CC0 |
| [ds002778](#ds002778-parkinsons-disease) | Parkinson's vs control | 31 (15 PD, 16 controls) | participant | 32 + 8 extra, 512 Hz | CC0 |
| [MPI-LEMON](#mpi-lemon-young-and-old-adults) | Young vs old, eyes closed / open | 202 (137 young, 65 old) | participant | 61 (preprocessed), 250 Hz | PDDL 1.0 |
| [BNCI2014_001](#bnci2014_001-bci-competition-iv-2a) | Left vs right hand imagery | 9 subjects, 2,592 trials | trial | 22 EEG + 3 EOG, 250 Hz | CC BY-ND 4.0 |
| [Sleep-EDF](#sleep-edf-expanded-sleep-cassette) | Sleep staging (5 stages) | 20 subjects, 39 nights, 5,767 epochs | 30 s epoch | 2 bipolar, 100 Hz | ODC-By 1.0 |

## Common pipeline

Unless a section says otherwise (`_clean` in `src/untrained_eeg/data.py`): pick the channels
listed in the config, average reference, band-pass 1–40 Hz (MNE's default FIR), resample to
128 Hz, convert to microvolts, cut into non-overlapping 2 s windows. Windows are grouped in
1-minute blocks or segments, and features are averaged per block (`scripts/phase1_features.py`).
Probe: standardized logistic regression, stratified 5-fold CV with a fixed split (`cv_seed` 0);
details per task in `src/untrained_eeg/tasks.py`.

Pretrained models get the input they were trained on, cached as variants of the same data
(`preprocessing` in the configs, `COMMON_VARIANTS` in `data.py`): 4–38 Hz at 250 Hz with
exponential moving standardization for the pretrained ShallowFBCSPNet, 200, 250 or 256 Hz for the
foundation models (30 s windows for CodeBrain). A checkpoint is never evaluated on a dataset it
was trained on (`exclude` in the configs, leakage table in `docs/pretrained_models.md`).

## ds003478: depression, rest

- **Source:** [OpenNeuro ds003478](https://openneuro.org/datasets/ds003478/versions/1.1.0),
  v1.1.0, "EEG: Depression rest", J. F. Cavanagh. DOI 10.18112/openneuro.ds003478.v1.1.0.
  Reference given by the dataset: Cavanagh et al. (2019), *Computational Psychiatry* 3:1–17,
  doi:10.1162/cpsy_a_00024. License CC0.
- **Participants:** 122 in the dataset. We keep 113: participant 544 is excluded
  (unstable BDI, as in the original study) and BDI 7–16 or missing is dropped. Low BDI ≤ 6
  (label 0, 74), high BDI ≥ 17 (label 1, 39). Age 18–24 (mean 18.9); 67 female, 46 male.
- **Recording:** Neuroscan SynAmps2, 64 channels, 500 Hz, recorded with a reference between Cz
  and CPz, 60 Hz line. Run-01 only (recorded right after cap hook-up); run-02 was never quality
  checked by the authors. Six 1-minute blocks per run, three eyes closed and three eyes open.
- **Our preprocessing:** the common pipeline on 60 scalp channels (M1, M2, CB1, CB2, HEOG, VEOG
  and an EKG channel present in some files are dropped). Blocks are found from the triggers
  (codes 1–6 every 500 ms, odd = eyes closed, even = eyes open); blocks shorter than 60 s are
  skipped. 650 complete blocks of 30 windows (1–3 per participant and condition).
- **Tasks:**
  - Eyes closed vs open (positive control): one sample per block, 650 blocks; folds grouped by
    participant.
  - High vs low BDI, eyes closed and eyes open separately: one sample per participant (mean of
    their blocks of that condition), 113 participants.
- **Quirks:** HEOG/VEOG may be swapped; some channels were interpolated by the authors and no raw
  data is available to undo it. Participant 516 has no run-02 (we use run-01). Groups differ in
  sex and trait anxiety (STAI), see issue #32.

## ds004504: Alzheimer's and frontotemporal dementia

- **Source:** [OpenNeuro ds004504](https://openneuro.org/datasets/ds004504), v1.0.9.
  DOI 10.18112/openneuro.ds004504.v1.0.9. License CC0. Cite the data descriptor: Miltiadous et
  al. (2023), *Data* 8(6):95, doi:10.3390/data8060095. The authors also published the AHEPA
  benchmark with a reporting checklist for studies using this dataset (doi:10.1007/s11571-026-10464-w).
- **Participants:** 88: 36 Alzheimer's disease (age 66.4 ± 7.9), 23 frontotemporal dementia
  (63.7 ± 8.2) and 29 healthy controls (67.9 ± 5.4). Diagnosis and MMSE from the 2nd Department
  of Neurology, AHEPA University Hospital, Thessaloniki.
- **Recording:** Nihon Kohden EEG 2100, 19 electrodes (10-20: Fp1, Fp2, F7, F3, Fz, F4, F8, T3,
  C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2), 500 Hz, reference A1–A2, 50 Hz line. Resting,
  eyes closed, seated, about 12–14 minutes per participant (5–21).
- **Our preprocessing:** the common pipeline on the raw recordings (`sub-0XX/eeg`; the authors'
  ASR + ICA derivatives are not used). Each continuous recording is cut into complete 60 s
  segments of 30 windows: 1,132 segments, 5–21 per participant.
- **Tasks:** one sample per participant, the mean of all their segments. AD vs control (65:
  36 vs 29) and FTD vs control (52: 23 vs 29). The other patient group is left out of each task.

## ds002778: Parkinson's disease

- **Source:** [OpenNeuro ds002778](https://openneuro.org/datasets/ds002778), v1.0.4, UC San
  Diego, curated by A. P. Rockhill. DOI 10.18112/openneuro.ds002778.v1.0.4. License CC0.
- **Before submitting a manuscript, email the curators** (arockhil@uoregon.edu), as the README
  asks. The README also warns against reporting high patient-vs-control accuracy from this dataset
  alone, without validation on another dataset.
- **Cite** (from the README): Jackson et al. (2019), *eNeuro* 6(3), doi:10.1523/ENEURO.0151-19.2019;
  Swann et al. (2015), *Ann Neurol* 78(5):742–750, doi:10.1002/ana.24507; George et al. (2013),
  *NeuroImage: Clinical* 3:261–270, doi:10.1016/j.nicl.2013.07.013.
- **Participants:** 31: 15 Parkinson's patients (age 63.3 ± 8.2; 8 female) and 16 controls
  (63.5 ± 9.7; 9 female). Patients have an off- and an on-medication session; we use the
  off-medication session (`sessions` in the config), controls their only session.
- **Recording:** BioSemi, 512 Hz, about 3 minutes at rest. The files have 40 channels; we keep
  the 32 scalp channels. Eyes open or closed is not stated in the dataset.
- **Our preprocessing:** the common pipeline, 60 s segments of 30 windows: 94 segments, 3–4 per
  participant.
- **Task:** Parkinson's vs control, one sample per participant (31). With n = 31, AUCs vary a
  lot between CV splits; report them with that spread.

## MPI-LEMON: young and old adults

- **Source:** [MPI Leipzig Mind-Brain-Body (LEMON)](https://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html),
  INDI public S3 bucket. License PDDL 1.0 (Open Data Commons Public Domain Dedication). Cite
  Babayan et al. (2019), *Scientific Data* 6:180308, doi:10.1038/sdata.2018.308.
- **Participants:** 228 in the study, 203 with preprocessed EEG on the server. Age is given in
  5-year bins. We keep 202: young 20–35 (bins 20-25, 25-30, 30-35; 137) and old 55–80 (bins
  55-60 to 75-80; 65). The one participant in the 35–40 bin is left out. Label 1 = old.
- **Recording used:** the authors' preprocessed EEG
  (`EEG_Preprocessed_BIDS_ID/EEG_Preprocessed`), one EEGLAB file per condition (eyes closed, eyes
  open), 250 Hz, about 8 minutes each, 56–61 channels after the authors removed bad channels.
  Two participants (sub-032483, sub-032484) were preprocessed at 100 Hz instead; like everyone
  they are resampled to 128 Hz, so their signal has nothing above 50 Hz.
  Each file joins the condition's blocks, marked by `boundary` events. The raw recordings
  (68 GB, BrainVision) are not used.
- **Our preprocessing:** missing channels are rebuilt by spherical-spline interpolation so that
  everyone has the same 61 channels, then the common pipeline. 60 s segments of 30 windows,
  6–8 per participant and condition (2,900 in all). Segments are cut from the joined file, so a
  segment can span a block boundary.
- **Tasks:** young vs old, eyes closed and eyes open separately: one sample per participant (mean
  of that condition's segments), 202 participants.

## BNCI2014_001: BCI Competition IV 2a

- **Source:** [BNCI Horizon 2020 data set 001-2014](http://bnci-horizon-2020.eu/database/data-sets)
  (MOABB `BNCI2014_001`), data set 2a of BCI Competition IV, Graz University of Technology.
  License CC BY-ND 4.0: do not redistribute modified data. Cite Tangermann et al. (2012),
  *Frontiers in Neuroscience* 6:55, doi:10.3389/fnins.2012.00055.
- **Participants:** 9 subjects, 2 sessions on different days (T and E), 288 trials per session:
  left hand, right hand, feet, tongue (72 each).
- **Recording:** 22 EEG + 3 EOG channels, 250 Hz.
- **Our preprocessing:** each run is cleaned as one recording with the common pipeline on the 22
  EEG channels. Imagery period 2–6 s after trial start (cue at 2 s) = two 2 s windows, averaged
  into one sample. Left and right hand trials of both sessions: 288 per subject, 2,592 in all.
- **Task:** left vs right hand, one sample per trial. Each subject gets its own stratified 5-fold
  CV (within subject), and the AUC is computed over all subjects' trials.
- **Leakage:** both pretrained ShallowFBCSPNet checkpoints (subject 3) and the EEGNetv4
  `BNCI2014001` checkpoint were trained on this dataset and are left out of this task.

## Sleep-EDF Expanded: sleep cassette

- **Source:** [PhysioNet Sleep-EDF Database Expanded](https://physionet.org/content/sleep-edfx/1.0.0/)
  v1.0.0, sleep cassette (SC) study, downloaded from PhysioNet's open-data S3 mirror.
  DOI 10.13026/C2X676. License ODC-By 1.0. Cite Kemp et al. (2000), *IEEE Trans Biomed Eng*
  47(9):1185–1194, doi:10.1109/10.867928, and the standard PhysioNet citation given on the page.
- **Participants:** the study has 153 nights of 78 healthy adults, recorded at home. We use the
  first 20 subjects (files SC400*–SC419*) and both of their nights where there are two: 39 nights
  (subject 13 has one).
- **Recording:** EEG Fpz-Cz and Pz-Oz (bipolar), 100 Hz. Hypnograms in 30 s epochs, scored with
  Rechtschaffen and Kales.
- **Our preprocessing:** no re-reference (already bipolar), band-pass 0.5–40 Hz (to keep the slow
  waves of deep sleep), native 100 Hz. Stages 3 and 4 are merged into N3 (AASM); movement and
  unscored epochs are dropped. Wake is kept only within 30 minutes of sleep. At most 30 epochs
  per stage and night, drawn with a fixed seed: 5,767 epochs.
- **Task:** 5 classes (W, N1, N2, N3, REM), one sample per 30 s epoch. Folds are grouped by
  subject, so both nights of a subject are in the same fold. Metrics: macro one-vs-rest AUC and
  balanced accuracy.
- **Models:** bipolar derivations have no electrode positions, so no foundation model and no
  EEGNetv4 (C3, Cz, C4) runs here.

## Other candidates (not used)

| Dataset | What | Why not (yet) |
|---|---|---|
| [ds003775](https://openneuro.org/datasets/ds003775) | Resting EEG (SRM), ~111 healthy adults, 64 channels | No clinical label |
| [TUH EEG Corpus](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/) | Clinical EEG, normal vs abnormal (TUAB) | Registration required; most foundation models were pretrained on it (leakage) |

## Adding a dataset

1. Check it is open (no registration) and not in any checkpoint's pretraining data
   (`docs/pretrained_models.md`).
2. Copy the closest config in `configs/` and set the download source, `loader`, channels and
   `tasks`.
3. Add a section here with the same headings: source, participants, recording, our
   preprocessing, task, and anything odd you found.
