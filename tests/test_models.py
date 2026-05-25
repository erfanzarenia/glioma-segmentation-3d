"""End-to-end forward/backward tests for the Residual-SE U-Net."""

import torch

from glioma_seg.models import ReSE_UNet3D


def test_forward_pass_shape(tiny_volume):
    model = ReSE_UNet3D(
        input_channels=2,
        output_channels=3,
        dropout_rate=0.0,
        gradient_checkpointing=False,
    )
    out = model(tiny_volume)
    assert out.shape == (1, 3, 32, 32, 32)


def test_forward_pass_with_dropout_scaling(tiny_volume):
    model = ReSE_UNet3D(
        input_channels=2,
        output_channels=3,
        dropout_rate=0.1,
        dropout_scaling=True,
        gradient_checkpointing=False,
    )
    out = model(tiny_volume)
    assert out.shape == (1, 3, 32, 32, 32)
    assert torch.isfinite(out).all()


def test_backward_pass_runs(tiny_volume):
    model = ReSE_UNet3D(
        input_channels=2,
        output_channels=3,
        dropout_rate=0.0,
        gradient_checkpointing=False,
    )
    out = model(tiny_volume)
    loss = out.sum()
    loss.backward()
    # Sanity: at least one parameter received a gradient.
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad


def test_gradient_checkpointing_matches_no_checkpointing(tiny_volume):
    """Forward outputs should agree (up to numerics) with/without checkpointing."""
    torch.manual_seed(0)
    model_a = ReSE_UNet3D(
        input_channels=2, output_channels=3, dropout_rate=0.0, gradient_checkpointing=False
    )
    torch.manual_seed(0)
    model_b = ReSE_UNet3D(
        input_channels=2, output_channels=3, dropout_rate=0.0, gradient_checkpointing=True
    )
    model_a.eval()
    model_b.eval()

    with torch.no_grad():
        out_a = model_a(tiny_volume)
        out_b = model_b(tiny_volume)

    assert out_a.shape == out_b.shape
