"""The shipped Hydra config builds every object the CLI scripts need."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from monai.losses import DiceFocalLoss, GeneralizedDiceFocalLoss
from omegaconf import OmegaConf

from glioma_seg.data import Augmenter
from glioma_seg.models import ReSE_UNet3D
from glioma_seg.training import build_loss
from glioma_seg.utils.device import resolve_device

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


@pytest.fixture(scope="module")
def cfg():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="config")


def test_default_config_builds_pipeline_objects(cfg, tiny_volume, tiny_label):
    resolve_device(cfg.device)
    Augmenter(**cfg.training.augmentation).compose()
    model = ReSE_UNet3D(
        input_channels=cfg.model.input_channels,
        output_channels=cfg.model.output_channels,
        dropout_rate=cfg.model.dropout_rate,
        dropout_scaling=cfg.model.dropout_scaling,
        gradient_checkpointing=False,
    )
    loss_fn = build_loss(cfg.training.loss)
    assert isinstance(loss_fn, GeneralizedDiceFocalLoss)
    loss = loss_fn(model(tiny_volume), tiny_label)
    assert torch.isfinite(loss)


def test_build_loss_dice_focal_and_unknown(cfg):
    loss_cfg = OmegaConf.merge(cfg.training.loss, {"name": "dice_focal"})
    assert isinstance(build_loss(loss_cfg), DiceFocalLoss)
    with pytest.raises(ValueError, match="Unknown loss"):
        build_loss(OmegaConf.merge(cfg.training.loss, {"name": "nope"}))
