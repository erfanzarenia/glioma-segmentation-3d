"""Tests for the training-time augmentation pipeline."""

import torch
from monai.transforms import RandAffined

from glioma_seg.data import Augmenter


def _rand_affine(compose):
    return next(t for t in compose.transforms if isinstance(t, RandAffined))


def test_affine_scale_is_within_ten_percent():
    affine = _rand_affine(Augmenter().compose())
    affine.set_random_state(0)
    grid = affine.rand_affine.rand_affine_grid
    for _ in range(200):
        grid.randomize()
        scales = torch.as_tensor(grid.scale_params)
        assert scales.numel() == 3  # all three spatial axes are scaled
        assert ((scales >= 0.9 - 1e-6) & (scales <= 1.1 + 1e-6)).all()


def test_augmented_label_keeps_original_classes():
    """Labels must be resampled with nearest-neighbour, never interpolated."""
    image = torch.randn(2, 32, 32, 32)
    label = torch.zeros(1, 32, 32, 32, dtype=torch.int64)
    label[:, 8:24, 8:24, 8:24] = 2
    label[:, 12:20, 12:20, 12:20] = 0  # 0/2 boundary: interpolation would create 1s

    compose = Augmenter(flip_prob=1.0, rotate_prob=1.0, affined_prob=1.0).compose()
    compose.set_random_state(0)
    for _ in range(5):
        out = compose({"image": image.clone(), "label": label.clone()})
        assert out["image"].shape == image.shape
        assert out["label"].shape == label.shape
        values = torch.unique(torch.as_tensor(out["label"]))
        assert set(values.tolist()) <= {0.0, 2.0}
