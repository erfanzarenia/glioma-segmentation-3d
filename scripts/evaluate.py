"""CLI entry point: evaluate a checkpoint on the test split.

Example
-------
    python scripts/evaluate.py
    python scripts/evaluate.py inference.checkpoint=path/to/your.pt
"""

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from glioma_seg.inference import Evaluator
from glioma_seg.models import ReSE_UNet3D
from glioma_seg.training import DataManager, build_loss
from glioma_seg.utils.logging import get_logger
from glioma_seg.utils.seed import set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log = get_logger()
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)

    dm = DataManager(
        processed_dir=cfg.data.processed_dir,
        batch_size=cfg.inference.batch_size,
        num_workers=cfg.inference.num_workers,
        transform=None,
    )
    test_loader = dm.get_test_loader()

    model = ReSE_UNet3D(
        input_channels=cfg.model.input_channels,
        output_channels=cfg.model.output_channels,
        dropout_rate=0.0,
        dropout_scaling=False,
        gradient_checkpointing=False,
    )

    ckpt_path = cfg.inference.checkpoint
    log.info("Loading checkpoint: %s", ckpt_path)
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)

    loss_fn = build_loss(cfg.training.loss)

    evaluator = Evaluator(model=model, loss_func=loss_fn, device=cfg.device)
    avg_loss, dice, hd95 = evaluator.evaluate(test_loader, average_metric=False)

    log.info("Test loss : %.4f", avg_loss)
    for name, d, h in zip(
        ("NE+edema (class 1)", "Enhancing (class 2)"), dice.tolist(), hd95.tolist(), strict=True
    ):
        log.info("%-20s Dice=%.4f  HD95=%.2f", name, d, h)


if __name__ == "__main__":
    main()
