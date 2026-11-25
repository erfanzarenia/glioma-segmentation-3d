# Experiments

Living document for ablations, baselines, and follow-up runs.

## Headline result

Bundled checkpoint `checkpoints/model_epoch150.pt` (single run, 150 epochs),
evaluated with `scripts/evaluate.py` (full-volume inference, MONAI metrics).

| Class | Test Dice ↑ | Test HD95 (mm) ↓ | Val Dice | Val HD95 (mm) |
|---|---|---|---|---|
| Non-enhancing + edema (merged) | 0.914 | 3.50 | 0.885 | 5.02 |
| Enhancing tumour | 0.923 | 3.63 | 0.827 | 4.69 |

Test = 126 subjects, val = 125. Enhancing-tumour scores exclude subjects with
no enhancing tumour in the ground truth (5 test, 26 val); see
[`METHODOLOGY.md`](METHODOLOGY.md#evaluation).

Notes:

- Earlier versions of this document reported Dice 0.890 / 0.830 and HD95
  4.16 / 4.33 as test-set results. The original notebook's test loader read
  the **validation** folder, and those Dice values match the validation split
  above; the HD95 values could not be reproduced.
- Two modalities (FLAIR + T1ce) only.
- Classes are **not** the BraTS standard ET / TC / WT (see
  [`METHODOLOGY.md`](METHODOLOGY.md)). Numbers above are not directly comparable
  to the BraTS leaderboard.

## Proposed ablations

| # | Variant | Hypothesis | Status |
|---|---|---|---|
| 1 | Plain U-Net (no residual, no SE) | Isolate the contribution of residual + SE | TODO |
| 2 | U-Net + residual, no SE | Isolate SE's contribution | TODO |
| 3 | Full Residual + SE (current) | Baseline of the repo | done |
| 4 | + adaptive-dropout off | Test whether depth-scaled dropout helps | TODO |
| 5 | + sliding-window inference | Quantify gap vs. full-volume eval | TODO |
| 6 | + connected-component post-processing | Reduce ET false positives | TODO |
| 7 | + all 4 modalities (T1, T1ce, T2, FLAIR) | Quantify cost of dropping T1 + T2 | TODO |
| 8 | nnU-Net (default config) | Strong baseline for comparability | TODO |

## Re-evaluation under standard BraTS regions

The model currently predicts `{background, NE+edema, ET}`. To produce
comparable ET / TC / WT scores from these predictions:

| BraTS region | Definition | Mapping from refined labels |
|---|---|---|
| WT (whole tumour) | All tumour voxels | `{1, 2}` |
| TC (tumour core) | Enhancing + necrotic core | `{2}` ∪ (necrotic core, which is currently merged with edema; see note below) |
| ET (enhancing tumour) | Enhancing only | `{2}` |

**WT and ET can be recovered exactly** from the current label mapping;
**TC cannot**, because the original BraTS class 1 (necrotic core) was merged
with class 2 (edema) into a single class in this pipeline. A faithful TC score
requires re-running preprocessing with the merge changed to
`{1 → 1, 2 → 2, 4 → 3}` (three foreground classes) and retraining with four
output classes. This is listed under *Future improvements* in the README; it
is not yet one of the ablations above.

## Loss used for the bundled checkpoint

The training history saved with the checkpoint logs a validation loss of
≈ 0.14 at epoch 150. Recomputing the validation loss of
`model_epoch150.pt` under candidate losses:

| Loss | Val loss |
|---|---|
| `GeneralizedDiceFocalLoss`, background included, `lambda_focal=0.2` (original notebook) | 0.151 |
| `GeneralizedDiceFocalLoss`, `lambda_focal=1.0`, with / without background | 0.157 |
| `DiceFocalLoss`, background included, `lambda_focal=1.0` | 0.163 |
| `DiceFocalLoss`, background excluded, `lambda_focal=1.0` | 0.238 |

The starting point agrees as well: training from scratch (4-subject smoke
run) gives an epoch-1 train / val loss of 1.02 / 1.02 under the default loss,
against 1.03 / 1.02 in the logged history, whereas `DiceFocalLoss` without
background starts at 1.16 / 1.16.

The Generalized Dice + Focal loss from the original notebook is the closest
match and is the default in `configs/training/default.yaml`. The match is not
exact (the history values are rounded, and the recomputed validation Dice is
86.5% against 85.8% logged), so the exact loss weights of the original
150-epoch run cannot be confirmed from the checkpoint alone.

## Logging

Recommended: add `wandb` (or TensorBoard) to track loss/Dice/lr per epoch and
a few qualitative plots per validation step.

## Hyperparameters tried (informal)

| Setting | Range explored | Best | Notes |
|---|---|---|---|
| Learning rate | `1e-4` (only) | `1e-4` | Inherited from MONAI BraTS examples |
| Dropout | `{0.0, 0.15}` | `0.15` | Depth-scaled (`×1.25` per level) |
| Augmentation strength | default MONAI knobs | default | Not formally ablated |

> If you ablate something, add a row here with `(epoch, dice, hd95)` so future
> runs are traceable.
