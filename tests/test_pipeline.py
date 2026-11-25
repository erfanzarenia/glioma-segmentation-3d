"""End-to-end test on a tiny synthetic BraTS-style dataset.

Raw NIfTI -> Preprocessor -> DataManager -> Trainer (1 epoch) -> Evaluator,
all on the CPU with small volumes.
"""

import math

import nibabel as nib
import numpy as np
import pytest
import torch
from monai.losses import DiceFocalLoss

from glioma_seg.data import Augmenter, Inspector, Preprocessor
from glioma_seg.inference import Evaluator
from glioma_seg.models import ReSE_UNet3D
from glioma_seg.training import DataManager, Trainer
from glioma_seg.utils.device import resolve_device

N_SUBJECTS = 10
SHAPE = (40, 40, 40)


def _write_subject(root, name, rng):
    subj = root / name
    subj.mkdir()
    brain = np.zeros(SHAPE, dtype=np.float32)
    brain[4:36, 4:36, 4:36] = 1.0
    seg = np.zeros(SHAPE, dtype=np.float32)
    seg[10:30, 10:30, 10:30] = 2  # edema
    seg[14:26, 14:26, 14:26] = 1  # necrotic core
    seg[17:23, 17:23, 17:23] = 4  # enhancing
    affine = np.eye(4)
    for modality in ("flair", "t1ce", "t1", "t2"):
        vol = brain * rng.uniform(100, 1000, SHAPE).astype(np.float32)
        nib.save(nib.Nifti1Image(vol, affine), subj / f"{name}_{modality}.nii.gz")
    nib.save(nib.Nifti1Image(seg, affine), subj / f"{name}_seg.nii.gz")


@pytest.fixture(scope="module")
def processed_dir(tmp_path_factory):
    rng = np.random.default_rng(0)
    raw = tmp_path_factory.mktemp("raw")
    for i in range(N_SUBJECTS):
        _write_subject(raw, f"BraTS2021_{i:05d}", rng)
    (raw / "name_mapping.csv").write_text("stray file that is not a subject\n")

    out = tmp_path_factory.mktemp("processed")
    pre = Preprocessor(str(raw), str(out), split_ratio=(0.8, 0.1, 0.1))
    crop, stats = pre.compute_metadata()
    assert crop == {"x": (4, 36), "y": (4, 36), "z": (4, 36)}
    assert all(math.isfinite(v) and v > 0 for v in stats.values())
    pre.run(crop, stats)
    return out


def test_preprocessing_split_shapes_and_labels(processed_dir):
    inspector = Inspector(str(processed_dir))
    expected = {"train": 8, "val": 1, "test": 1}
    for split, count in expected.items():
        n_data, n_labels, bad_data, bad_labels = inspector.inspect(
            split, (2, 32, 32, 32), torch.float32, (1, 32, 32, 32), torch.int64
        )
        assert (n_data, n_labels) == (count, count)
        assert not bad_data
        assert not bad_labels

    label = torch.load(next((processed_dir / "train" / "labels").iterdir()))
    assert set(torch.unique(label).tolist()) == {0, 1, 2}


def test_train_and_evaluate_one_epoch(processed_dir):
    torch.manual_seed(0)
    dm = DataManager(
        processed_dir=str(processed_dir),
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        transform=Augmenter().compose(),
    )
    model = ReSE_UNet3D(input_channels=2, output_channels=3, gradient_checkpointing=True)
    loss_fn = DiceFocalLoss(include_background=False, to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in model.parameters()]

    trainer = Trainer(
        model, optimizer, None, loss_fn, accumulation_steps=4, device="cpu", amp=False
    )
    train_loss = trainer.train(dm.get_train_loader())
    assert math.isfinite(train_loss)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))

    evaluator = Evaluator(model, loss_fn, device="cpu")
    val_loss, dice, _ = evaluator.evaluate(dm.get_val_loader(), average_metric=True)
    assert math.isfinite(val_loss)
    assert dice.numel() == 1

    _, dice, hd95 = evaluator.evaluate(dm.get_test_loader(), average_metric=False)
    assert dice.shape == (2,)
    assert hd95.shape == (2,)


def test_resolve_device():
    assert resolve_device("cpu") == torch.device("cpu")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto").type == expected
    assert resolve_device(None).type == expected
