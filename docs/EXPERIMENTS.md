# Experiments

Living document for ablations, baselines, and follow-up runs.

## Headline result

Held-out test split (10% of BraTS 2021), single-run, 150 epochs.

| Class | Dice ↑ | HD95 (mm) ↓ |
|---|---|---|
| Non-enhancing + edema (merged) | 0.890 | 4.16 |
| Enhancing tumour | 0.830 | 4.33 |

Notes:

- Two modalities (FLAIR + T1ce) only.
- Classes are **not** the BraTS standard ET / TC / WT — see
  [`METHODOLOGY.md`](METHODOLOGY.md). Numbers above are not directly comparable
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
| TC (tumour core) | Enhancing + necrotic core | `{2}` ∪ (necrotic core, which is currently merged with edema — see note below) |
| ET (enhancing tumour) | Enhancing only | `{2}` |

**WT and ET can be recovered exactly** from the current label mapping;
**TC cannot**, because the original BraTS class 1 (necrotic core) was merged
with class 2 (edema) into a single class in this pipeline. A faithful TC score
requires re-running preprocessing with the merge changed to
`{1 → 1, 2 → 2, 4 → 3}` (three foreground classes) — tracked as
[`ablation 7`](#proposed-ablations).

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
