# Methodology

The description of the data, model, training, and evaluation.

## Dataset

[BraTS 2021](http://www.braintumorsegmentation.org/) provides **1 251**
multi-modal 3D MRI volumes (T1, T1ce, T2, FLAIR) with voxel-level segmentation
labels:

| Original label | Meaning |
|---|---|
| 0 | Background |
| 1 | Necrotic / non-enhancing tumour core |
| 2 | Peritumoral edema |
| 4 | Enhancing tumour |

## Modality and label choices

To fit a 3D U-Net into a single consumer GPU under the experimental setup, only
**FLAIR** and **T1ce** are used. Labels are merged to three classes:

| Refined label | Source | Meaning |
|---|---|---|
| 0 | BraTS 0 | Background |
| 1 | BraTS {1, 2} | Non-enhancing core + edema |
| 2 | BraTS 4 | Enhancing tumour |

**Caveat.** This is *not* the standard BraTS ET / TC / WT formulation. Numbers
in this repository are therefore not directly comparable to BraTS leaderboard
entries. See [`EXPERIMENTS.md`](EXPERIMENTS.md) for a planned re-evaluation
using standard regions.

## Preprocessing pipeline

Implemented in [`glioma_seg.data.preprocessing.Preprocessor`](../src/glioma_seg/data/preprocessing.py).

1. **Scan all subjects once** to compute:
   - global non-zero bounding box, rounded to the nearest multiple of 8,
   - per-modality mean and std over non-zero voxels.
2. **For each subject**:
   - crop all volumes to the common bbox,
   - z-score normalise FLAIR and T1ce,
   - apply `refine_label`,
   - save as PyTorch tensors under `data/processed/{train,val,test}/{data,labels}`.

The split is **deterministic and sorted** (not randomised): subjects are
assigned to train/val/test by their sorted-index position. See
`Preprocessor.run`.

## Augmentation

Implemented in [`glioma_seg.data.transforms.Augmenter`](../src/glioma_seg/data/transforms.py),
applied online during training only.

| Transform | Probability | Parameters |
|---|---|---|
| `RandFlipd` (axes 0, 1, 2) | 0.5 each | — |
| `RandRotate90d` (planes 01, 12, 02) | 0.5 each | — |
| `RandAffined` | 0.3 | rotate ±10°, scale ±10%, translate ±10 vox |
| `RandScaleIntensityd` | 0.5 | factors ±0.1 |
| `RandShiftIntensityd` | 0.5 | offsets ±0.1 |
| `RandGaussianNoised` | 0.2 | std 0.1 |
| `RandBiasFieldd` | 0.3 | degree 2, coeff (0, 0.1) |

## Architecture

Implemented in [`glioma_seg.models.resunet_se.ReSE_UNet3D`](../src/glioma_seg/models/resunet_se.py).

A 3-stage encoder + bottleneck + 3-stage decoder, with every block being a
**Residual + Squeeze-and-Excitation block** built from:

```
Conv3D → InstanceNorm → ReLU → Conv3D → InstanceNorm → ReLU → SE → (+ Residual)
```

| Stage | In ch | Out ch | Kernel | Residual |
|---|---|---|---|---|
| `encod1` | 2 | (32, 64) | (5, 3) | no |
| `pool1` | 64 | 64 | 2 (max) | — |
| `encod2` | 64 | (64, 128) | (3, 3) | no |
| `pool2` | 128 | 128 | 2 (max) | — |
| `encod3` | 128 | (128, 256) | (3, 3) | **yes** |
| `pool3` | 256 | 256 | 2 (max) | — |
| `bottleneck` | 256 | (256, 256) | (3, 3) | **yes** |
| `upsamp3` | 256 → 128 | — | 2 (trans-conv) | — |
| `decod3` | 256+128 | (128, 64) | (3, 3) | **yes** |
| `upsamp2` | 64 → 32 | — | 2 (trans-conv) | — |
| `decod2` | 128+32 | (64, 32) | (3, 3) | no |
| `upsamp1` | 32 → 16 | — | 2 (trans-conv) | — |
| `decod1` | 64+16 | (64, 32) | (3, 5) | no |
| `outconv` | 32 | 3 | 1×1×1 | — |

Channel-attention reduction in `SE_Block` is `channels // 8`, so the bottleneck
SE uses a 32-channel bottleneck. Dropout scales by depth as
`{p, 1.25p, 1.25²p, 1.25³p}` for encoder stages 1–3 and the bottleneck.

Gradient checkpointing is wrapped per-block on encoder stages 2 + 3, the
bottleneck, and decoder stages 2 + 3. The first encoder and final decoder
stages are not checkpointed because their activations dominate VRAM less than
the deeper blocks.

## Loss

`monai.losses.DiceFocalLoss` with softmax over 3 classes, `include_background=False`,
`lambda_dice=1.0`, `lambda_focal=1.0`, `gamma=2.0`.

## Optimisation

| | Value |
|---|---|
| Optimizer | `AdamW`, lr `1e-4`, weight decay `1e-5` |
| Scheduler | `ReduceLROnPlateau` (patience 12, factor 0.5, min lr `1e-6`) |
| Epochs | 150 |
| Batch | 1 |
| Grad accumulation | 8 (effective batch 8) |
| Mixed precision | `torch.amp.autocast` + `GradScaler` |
| Grad checkpointing | on (see model spec) |

## Evaluation

Implemented in [`glioma_seg.inference.evaluator.Evaluator`](../src/glioma_seg/inference/evaluator.py).

- `monai.metrics.DiceMetric` with `include_background=False`.
- `monai.metrics.HausdorffDistanceMetric` with `percentile=95`,
  `include_background=False`.
- Predictions are taken as `argmax` over the logits and one-hot encoded for
  metric aggregation.

## Known limitations

- **Two modalities only** (FLAIR + T1ce); the full four-modality protocol is
  expected to help, particularly for the enhancing-tumour class.
- **Custom labels** (not ET/TC/WT), so scores are not directly comparable to
  the BraTS leaderboard.
- **No post-processing** (e.g. connected-component filtering) for fragmented
  enhancing-tumour predictions.
- **Sorted (not randomised) train/val/test split.**

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for proposed follow-ups.
