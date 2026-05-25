"""Unit tests for preprocessing utility functions."""

import numpy as np

from glioma_seg.data.preprocessing import (
    calculate_crop_range,
    refine_label,
    volume_crop,
    z_score,
)


def test_calculate_crop_range_rounds_to_multiple_of_eight():
    cx, cy, cz = calculate_crop_range(0, 10, 0, 11, 0, 17)
    for lo, hi in (cx, cy, cz):
        size = hi - lo
        assert size % 8 == 0


def test_volume_crop_returns_expected_shape():
    vol = np.zeros((32, 32, 32))
    cropped = volume_crop(vol, (4, 20), (8, 24), (0, 16))
    assert cropped.shape == (16, 16, 16)


def test_z_score_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=5.0, scale=2.0, size=10_000)
    mean = data.mean()
    std = data.std()
    z = z_score(data, mean, std)
    assert abs(z.mean()) < 1e-6
    assert abs(z.std() - 1.0) < 1e-6


def test_refine_label_merges_brats_classes():
    raw = np.array([[0, 1, 2, 4]], dtype=np.float32)
    refined = refine_label(raw)
    assert refined.dtype == np.int64
    assert refined.tolist() == [[0, 1, 1, 2]]
