# Brain Glioma Segmentation: Residual-SE U-Net (3D, Multi-Modal MRI)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.3+](https://img.shields.io/badge/pytorch-2.3+-ee4c2c.svg)](https://pytorch.org/)

Fully automated 3D segmentation of brain gliomas from multi-modal MRI, using a
U-Net augmented with residual connections and Squeeze-and-Excitation blocks.
Trained on BraTS 2021 with FLAIR and T1ce modalities under single-GPU memory
constraints.

![Test-set results](assets/qualitative_results.png)

## Overview

Gliomas are the most common primary brain tumour. The BraTS dataset annotates
three nested sub-regions: **enhancing tumour (ET)**, **tumour core (TC)**, and
**whole tumour (WT)**. Manual delineation is slow and subject to inter-rater
variability. This project implements an end-to-end 3D deep-learning pipeline
that segments gliomas from preprocessed MRI volumes, with a focus on the
engineering trade-offs required to train a 3D U-Net on a single consumer GPU.

## Motivation

- Reproduce a competitive BraTS segmentation pipeline end-to-end in PyTorch.
- Explore the engineering trade-offs (mixed precision, gradient checkpointing,
  gradient accumulation) required to train 3D U-Nets on a single consumer GPU.
- Quantify the cost of restricting input to two modalities (FLAIR + T1ce)
  versus the full four-modality BraTS protocol.

## Architecture

A 3D U-Net with:

- **Residual + Squeeze-and-Excitation blocks** at every encoder/decoder stage
  for stable gradient flow and channel-wise feature recalibration.
- **Instance normalization** (preferred over BatchNorm for batch size 1).
- **MaxPool** for downsampling and **transposed convolution** for upsampling
  via skip connections.
- **Adaptive dropout** that scales with network depth.
- **Gradient checkpointing** on deeper encoder, bottleneck, and decoder
  stages to bound peak VRAM.

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the long-form description.

![Architecture](assets/architecture.png)

| | Value |
|---|---|
| Input | 2 channels (FLAIR, T1ce), full cropped volume |
| Output | 3 classes (background, merged non-enhancing+edema, enhancing) |
| Encoder depth | 3 stages (64 → 128 → 256 channels) + 256-channel bottleneck |
| Loss | Generalized Dice + Focal |
| Normalization | Instance |

## Preprocessing pipeline

1. Load BraTS 2021 NIfTI volumes (FLAIR, T1ce, segmentation).
2. Compute global non-zero crop bounds across the dataset; round to a multiple
   of 8 to match the U-Net's three pooling stages. On the 1 251 BraTS 2021
   training subjects this gives a 176 × 216 × 152 volume.
3. Per-modality **z-score normalization** using statistics computed from
   non-zero voxels across the whole dataset.
4. Refine BraTS labels: `{1, 2} → 1` (non-enhancing + edema merged),
   `{4} → 2` (enhancing tumour).
5. Augmentation (training only, via MONAI): random flips, 90° rotations, slight
   affine (±10° / ±10% / ±10 voxels), intensity scale/shift, Gaussian noise,
   bias-field simulation.

## Training setup

| | |
|---|---|
| Hardware | 1× NVIDIA RTX A4000 (16 GB VRAM) |
| Framework | PyTorch 2.6 + MONAI 1.4 (code also tested with PyTorch 2.3 + MONAI 1.3) |
| Optimizer | AdamW, lr 1e-4 |
| Scheduler | `ReduceLROnPlateau` (patience 12, factor 0.5) |
| Loss | Generalized Dice + Focal (MONAI `GeneralizedDiceFocalLoss`, background included, focal weight 0.2) |
| Batch | 1 (effective 8 via gradient accumulation) |
| Precision | Mixed (`torch.amp.autocast` + `GradScaler`) |
| Memory | Gradient checkpointing on encoder stages 2–3, bottleneck, decoder stages 2–3 |
| Dropout | 0.15, scaled ×1.25 per depth level |
| Epochs | 150 |
| Split | 80 / 10 / 10 train / val / test by sorted subject ID (1 000 / 125 / 126 subjects) |

## Results

The bundled checkpoint (`checkpoints/model_epoch150.pt`) evaluated with
`python scripts/evaluate.py` on the held-out **test** split (126 subjects).
The **validation** split (125 subjects) steered the learning-rate schedule
during training, so it is reported separately.

| Class | Test Dice ↑ | Test HD95 (mm) ↓ | Val Dice | Val HD95 (mm) |
|---|---|---|---|---|
| Non-enhancing + edema (merged) | **0.91** | **3.50** | 0.89 | 5.02 |
| Enhancing tumour | **0.92** | **3.63** | 0.83 | 4.69 |

**Important caveats.**

- Class definitions differ from the standard BraTS ET / TC / WT regions; the
  numbers above are **not directly comparable** to BraTS leaderboard entries.
- Evaluation is on a local split of the BraTS 2021 training data, not the
  official BraTS validation server.
- Subjects with no enhancing tumour in the ground truth (5 in test, 26 in
  val) are left out of the enhancing-tumour Dice and HD95, because MONAI
  scores an empty ground truth as NaN. False-positive enhancing voxels on those
  subjects (2 of the 5 test subjects, 22 of the 26 val subjects) are therefore
  not penalised; official BraTS scoring would count them.
- Only two modalities (FLAIR + T1ce) were used; the full four-modality protocol
  is expected to improve enhancing-tumour Dice in particular.

![Training and validation loss](assets/training_loss.png)
![Validation Dice score](assets/training_dice.png)

## Qualitative results

Axial mid-slice of three held-out subjects, FLAIR vs. ground truth vs.
prediction. These come from the validation split (the original notebook's
test loader read the validation folder).

![Sample predictions](assets/sample_predictions.png)

## Installation

Requires Python 3.10–3.12.

```bash
git clone https://github.com/erfanzarenia/glioma-segmentation-3d.git
cd glioma-segmentation-3d
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest    # unit tests + a small synthetic end-to-end run, CPU only
```

## Usage

All commands are run from the repository root. Settings live in `configs/`
and can be overridden on the command line (Hydra syntax, e.g. `data.raw_dir=...`).
Each run writes its resolved config and log to `runs/<timestamp>/`.

**Data.** BraTS 2021 must be requested from the
[official source](http://www.braintumorsegmentation.org/); it cannot be
redistributed. Place it at `data/raw/BraTS2021/` (one folder per subject).

**Preprocess**

```bash
python scripts/preprocess.py \
    data.raw_dir=data/raw/BraTS2021 \
    data.processed_dir=data/processed
```

This reads every subject twice (statistics, then cropping), takes about
20 minutes, and writes about **116 GB (108 GiB)** of `.pt` tensors for the
full 1 251-subject dataset: ≈ 92 MB per subject, half of it the `int64`
label volumes.

**Train**

```bash
python scripts/train.py
# override any config from CLI
python scripts/train.py training.epochs=200 model.dropout_rate=0.1
```

Checkpoints (`best.pt` by validation Dice, `last.pt`) and `history.pt` are
written to the run directory. The device defaults to CUDA when available; use
`device=cpu` to force the CPU, and `training.amp=false` to turn off mixed
precision (on the CPU, bf16 autocast is usually slower than fp32).

**Evaluate the bundled checkpoint**

The reference checkpoint (`checkpoints/model_epoch150.pt`, ~30 MB) is included
in the repo and used by default.

```bash
python scripts/evaluate.py
# or point at a different checkpoint
python scripts/evaluate.py inference.checkpoint=path/to/your.pt
```

It reports the test loss and per-class Dice / HD95 on the test split. The
whole 176 × 216 × 152 volume goes through the network in one pass; with
`device=cpu` this takes about 15 s per subject (≈ 30 min for the test split
on a 20-core machine).

## Repository layout

```
glioma-segmentation-3d/
├── configs/              # Hydra configs (model, data, training, inference)
├── src/glioma_seg/       # Importable package
│   ├── data/             # Dataset, transforms, preprocessing, I/O
│   ├── models/           # ReSE-UNet3D and building blocks
│   ├── training/         # Trainer, data manager, loss builder
│   ├── inference/        # Evaluator
│   └── utils/            # Visualization, seeding, logging, device selection
├── scripts/              # Thin CLI entry points (preprocess, train, evaluate)
├── tests/                # Unit tests + synthetic end-to-end pipeline test
├── assets/               # Figures used in README
├── checkpoints/          # Reference trained model
└── docs/                 # Methodology, experiment notes
```

## Future improvements

- Re-evaluate against the standard BraTS ET / TC / WT regions for comparability.
- Comparison against the **nnU-Net** baseline.
- Sliding-window inference and test-time augmentation.
- Train with all four modalities (T1, T1ce, T2, FLAIR).
- Connected-component post-processing for fragmented enhancing-tumour predictions.
- Lightweight Gradio demo + Hugging Face Space.

## Citation

If you reference this code, please cite:

```bibtex
@misc{zarenia2025glioma,
  author = {Zarenia, Erfan},
  title  = {Residual-SE U-Net for 3D Brain Glioma Segmentation},
  year   = {2025},
  howpublished = {\url{https://github.com/erfanzarenia/glioma-segmentation-3d}}
}
```

Underlying dataset:

```bibtex
@misc{baid2021rsna,
  author = {Baid, Ujjwal and others},
  title  = {The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor
            Segmentation and Radiogenomic Classification},
  year   = {2021},
  eprint = {2107.02314},
  archivePrefix = {arXiv}
}
```

## Acknowledgements

- The [MONAI](https://monai.io/) project for medical-imaging primitives.
- BraTS 2021 organizers and data contributors.
- Supervisor: Dr. J. Bruce Morton, Western University.

## License

MIT. See [LICENSE](LICENSE).
