"""Shared pytest fixtures."""

import pytest
import torch


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Always test on CPU in CI; the network is small enough for shape tests."""
    return torch.device("cpu")


@pytest.fixture
def tiny_volume() -> torch.Tensor:
    """A synthetic 2-channel volume sized for the U-Net's 3-stage downsampling.

    Shape is ``(1, 2, 32, 32, 32)`` — divisible by 8 so all pooling/upsampling
    stages line up.
    """
    return torch.randn(1, 2, 32, 32, 32)


@pytest.fixture
def tiny_label() -> torch.Tensor:
    """Integer label volume with classes ``{0, 1, 2}``, shape ``(1, 1, 32, 32, 32)``."""
    return torch.randint(low=0, high=3, size=(1, 1, 32, 32, 32))
