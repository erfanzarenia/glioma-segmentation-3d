"""Shape tests for SE and Residual-SE building blocks."""

import torch

from glioma_seg.models.blocks import Residual_SE_Block3D, SE_Block


def test_se_block_preserves_shape():
    x = torch.randn(1, 32, 8, 8, 8)
    block = SE_Block(channels=32, reduction_factor=8)
    out = block(x)
    assert out.shape == x.shape


def test_residual_se_block_output_channels():
    x = torch.randn(1, 16, 8, 8, 8)
    block = Residual_SE_Block3D(input_channels=16, output_channels=[32, 64], use_residual=True)
    out = block(x)
    assert out.shape == (1, 64, 8, 8, 8)


def test_residual_se_block_no_residual_path():
    x = torch.randn(1, 16, 8, 8, 8)
    block = Residual_SE_Block3D(input_channels=16, output_channels=[32, 64], use_residual=False)
    out = block(x)
    assert out.shape == (1, 64, 8, 8, 8)


def test_residual_se_block_same_channels_residual():
    """When input == output channels the 1×1×1 projection should be skipped."""
    x = torch.randn(1, 64, 8, 8, 8)
    block = Residual_SE_Block3D(input_channels=64, output_channels=[64, 64], use_residual=True)
    assert block.residual is None
    out = block(x)
    assert out.shape == x.shape
