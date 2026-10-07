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
