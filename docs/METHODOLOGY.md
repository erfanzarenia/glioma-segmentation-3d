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

On the 1 251 BraTS 2021 training subjects (240 × 240 × 155 each) this gives:

| | Value |
|---|---|
| Crop (x, y, z) | `[37, 213)`, `[10, 226)`, `[0, 152)` → 176 × 216 × 152 |
| FLAIR mean / std | 619.6 / 2772.3 |
| T1ce mean / std | 1522.6 / 13249.8 |

Rounding to a multiple of 8 can trim a few edge slices, but no tumour voxel
falls outside this crop. Background voxels (0 in the raw scans) are
normalised too, so they become `-mean / std` rather than 0.

The split is **deterministic and sorted** (not randomised): subjects are
assigned to train/val/test by their position in the sorted list of subject
folders (1 000 / 125 / 126). See `Preprocessor.run`. One side effect: 33
subjects have no enhancing tumour (no label 4), and 26 of them land in the
validation split, 5 in test and 2 in train.

## Augmentation

Implemented in [`glioma_seg.data.transforms.Augmenter`](../src/glioma_seg/data/transforms.py),
applied online during training only.

| Transform | Probability | Parameters |
|---|---|---|
| `RandFlipd` (axes 0, 1, 2) | 0.5 each | - |
| `RandRotate90d` (planes 01, 12, 02) | 0.5 each | - |
| `RandAffined` | 0.3 | rotate ±10°, scale ±10%, translate ±10 vox; label resampled nearest-neighbour |
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
| `pool1` | 64 | 64 | 2 (max) | - |
| `encod2` | 64 | (64, 128) | (3, 3) | no |
| `pool2` | 128 | 128 | 2 (max) | - |
| `encod3` | 128 | (128, 256) | (3, 3) | **yes** |
| `pool3` | 256 | 256 | 2 (max) | - |
| `bottleneck` | 256 | (256, 256) | (3, 3) | **yes** |
| `upsamp3` | 256 → 128 | - | 2 (trans-conv) | - |
| `decod3` | 256+128 | (128, 64) | (3, 3) | **yes** |
| `upsamp2` | 64 → 32 | - | 2 (trans-conv) | - |
| `decod2` | 128+32 | (64, 32) | (3, 3) | no |
| `upsamp1` | 32 → 16 | - | 2 (trans-conv) | - |
| `decod1` | 64+16 | (64, 32) | (3, 5) | no |
| `outconv` | 32 | 3 | 1×1×1 | - |

Each block's SE is built with `reduction_factor = C // 8` (where `C` is the
block's output channels), so the squeeze layer always has `C // (C // 8) = 8`
channels: 8 in every block, including the 256-channel bottleneck. Dropout
scales by depth as `{p, 1.25p, 1.25²p, 1.25³p}` for encoder stages 1–3 and
the bottleneck; decoder stages 3, 2 and 1 reuse the encoder rates of the same
depth.

Gradient checkpointing is wrapped per-block on encoder stages 2 + 3, the
bottleneck, and decoder stages 2 + 3. The first encoder and final decoder
stages run at full resolution and are not checkpointed: they hold the largest
activations, so checkpointing them would save the most memory but also cost
the most recomputation.

## Loss

`monai.losses.GeneralizedDiceFocalLoss` with softmax over 3 classes,
`include_background=True`, `lambda_gdl=1.0`, `lambda_focal=0.2`, `gamma=2.0`
(config block `training.loss`), as in the original training notebook. Set
`training.loss.name=dice_focal` to use `monai.losses.DiceFocalLoss` with the
same weights instead. See [`EXPERIMENTS.md`](EXPERIMENTS.md#loss-used-for-the-bundled-checkpoint)
for how this was checked against the bundled checkpoint.

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
- Inference runs on the whole cropped volume in one pass (no sliding window,
  no test-time augmentation).
- HD95 is computed in voxel units; BraTS volumes are 1 mm isotropic, so these
  are millimetres.
- Per-class scores are averaged over subjects, skipping NaNs. MONAI returns NaN
  for a class that is absent from a subject's ground truth, so subjects with
  no enhancing tumour do not count towards the enhancing-tumour Dice or HD95,
  even if the model predicts enhancing voxels there.

## Known limitations

- **Two modalities only** (FLAIR + T1ce); the full four-modality protocol is
  expected to help, particularly for the enhancing-tumour class.
- **Custom labels** (not ET/TC/WT), so scores are not directly comparable to
  the BraTS leaderboard.
- **No post-processing** (e.g. connected-component filtering) for fragmented
  enhancing-tumour predictions.
- **Sorted (not randomised) train/val/test split**, which also puts most
  subjects without enhancing tumour into the validation set (see above).

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for proposed follow-ups.
