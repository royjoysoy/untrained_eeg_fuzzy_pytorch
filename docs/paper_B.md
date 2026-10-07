# Paper analyses, group B: results that make it a paper (#31)

Builds on group A (`docs/paper_A.md`): same GPU extraction, probe and statistics scripts,
outputs under `results/paper/B*`. GPU (Turbulence op mode on the GB10), weight seed 0,
10 stochastic-rounding samples per setting, all windows.

## B5. Precision sweep

```bash
for t in 17 11 9 8; do
  python scripts/extract_gpu_features.py --config configs/paper.yaml --weight-seeds 0 --sr-seeds 0-9 \
      --precision $t --out results/paper/B5/p$t
  python scripts/probe_runs.py --config configs/paper.yaml --features results/paper/B5/p$t --out results/paper/B5/p${t}_probe
done
```

`--precision t` compiles `scripts/backends/srp.cu`, a Turbulence backend that stochastically
rounds every float32 add/sub/mul/div/fma to t significant bits (t = mantissa bits + 1:
24 = float32, 17, 11 = fp16-like, 9, 8 = bf16-like). The exact result is rounded to one of
its two neighbours on the t-bit grid with probabilities that make it unbiased, as
Verificarlo's SR at virtual precision t. Checks: at t = 24 the backend reproduces
Turbulence's own float32 rounding exactly (same spread to the last digit); at t = 8 the
spread of the features grows by ~2^16, as expected. t = 24 reuses the A1 samples.

## B6. One layer at a time

```bash
python scripts/extract_gpu_features.py --config configs/paper.yaml --models eegnet --weight-seeds 0 \
    --sr-seeds 0-9 --round-layer conv_spatial_depthwise --out results/paper/B6/eegnet_conv_spatial_depthwise
```

`--round-layer` splits the model into consecutive stages (`src/untrained_eeg/layers.py`;
applying them in order reproduces the model exactly) and compiles three GPU programs:
the stages before (round to nearest), the chosen one (stochastic rounding), the stages after
(round to nearest). Stages: 1D CNN conv1, elu1, pool1, conv2, elu2, global_pool;
ShallowFBCSPNet conv_time_spat, batch_norm, square, pool, log; EEGNet conv_temporal, bn1,
conv_spatial_depthwise, bn2, elu1, pool1, conv_separable_depth, conv_separable_point, bn3,
elu2, pool2 (dropout layers are identities in eval mode).

## B7. Seed ensembles

```bash
python scripts/seed_ensembles.py --config configs/paper.yaml --features results/paper/gpu_features \
    --out results/paper/B7 --k 1 2 5 10 20 --draws 50
```

For K random CNNs (50 draws of K distinct seeds out of the 30 A1 seeds) the features or the
per-seed predictions are averaged. With 30 seeds, draws of K = 20 share most of their seeds,
which lowers their spread beyond what independent ensembles would show; treat K = 20 as indicative.

## Summary

```bash
python scripts/summarize_b.py --seed-probe results/paper/A1 \
    --precision 24=results/paper/A1 17=results/paper/B5/p17_probe 11=results/paper/B5/p11_probe \
                9=results/paper/B5/p9_probe 8=results/paper/B5/p8_probe \
    --layers results/paper/B6/*_probe --out results/paper/B_summary
```

`precision_sweep.png`: median per-participant SD of P(high BDI) vs precision, per model and
condition, with the seed SD as a dashed reference line.

## Results (2026-10-07, all windows, weight seed 0, sklearn probe as in group A)

**B5.** Rounding variability grows ~10× per 3 bits removed but never reaches seed variability:
median per-participant SD of P(high BDI), as a fraction of the seed SD.

| model, condition | 24 bits (float32) | 17 | 11 (fp16-like) | 9 | 8 (bf16-like) | flips at 8 bits |
|---|---|---|---|---|---|---|
| cnn1d closed | 1.1e-6 | 2.3e-4 | 0.004 | 0.018 | 0.036 | 2.7 % |
| cnn1d open | 1.2e-6 | 1.3e-4 | 0.006 | 0.027 | 0.051 | 0 % |
| shallow closed | 5.0e-4 | 9.9e-4 | 0.011 | 0.048 | 0.092 | 7.1 % |
| shallow open | 5.6e-4 | 8.1e-4 | 0.012 | 0.049 | 0.091 | 10.6 % |
| eegnet closed | 2.9e-6 | 3.6e-4 | 0.003 | 0.009 | 0.020 | 2.7 % |
| eegnet open | 6.2e-7 | 1.3e-4 | 0.004 | 0.014 | 0.028 | 2.7 % |

Predictions start to flip at 11 bits and below (up to 4 % of participants at 11 bits, 11 %
at 8 bits). ShallowFBCSPNet stays flat near 1e-4 from 24 to 17 bits: that floor is the
probe's stopping tolerance amplifying tiny feature differences (see group A, task 4).

**B6.** Where the rounding noise enters (median per-participant SD, eyes closed / open):

- ShallowFBCSPNet: the merged temporal-spatial convolution dominates (5.9e-5 / 9.7e-5, close
  to the whole model's 1.1e-4); then pooling (3.1e-5 / 3.7e-6), batch norm (3.5e-6 / 4.7e-5),
  square (1.2e-6 / 2.8e-5). The log layer shows no effect (~1e-18): Turbulence's op mode does
  not seem to round inside `log`, so this is a tool limitation, not a result.
- EEGNet: every stage contributes a similar ~1e-7 (temporal conv largest, 2.5e-7; ELUs
  smallest, 2e-8): no single layer stands out.
- 1D CNN: the first convolution dominates (1.8e-7), ELUs contribute least (2e-8).
- No layer alone flips any prediction.

**B7.** Averaging the predictions of K = 20 random CNNs instead of using one:

| model, condition | AUC, K=1 → 20 | median SD, K=1 → 20 | flips, K=1 → 20 |
|---|---|---|---|
| cnn1d closed | 0.636 → 0.676 | 0.150 → 0.019 | 69 % → 12 % |
| cnn1d open | 0.449 → 0.436 | 0.032 → 0.004 | 30 % → 2 % |
| shallow closed | 0.562 → 0.589 | 0.201 → 0.028 | 83 % → 17 % |
| shallow open | 0.584 → 0.614 | 0.196 → 0.027 | 89 % → 14 % |
| eegnet closed | 0.536 → 0.578 | 0.246 → 0.030 | 87 % → 20 % |
| eegnet open | 0.562 → 0.596 | 0.108 → 0.015 | 73 % → 10 % |

Averaging predictions shrinks the seed spread ~7× and gains 0.03–0.04 AUC in five of six
cases; averaging features shrinks the spread less (`results/paper/B7/summary.csv`). K = 20 draws
overlap heavily (30 seeds), so their spread is a lower bound.
