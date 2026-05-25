"""CLI entry point: train the Residual-SE U-Net.

Example
-------
    python scripts/train.py
    python scripts/train.py training.epochs=200 model.dropout_rate=0.1
"""

import os

import hydra
import torch
from monai.losses import DiceFocalLoss
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from glioma_seg.data import Augmenter
from glioma_seg.inference import Evaluator
from glioma_seg.models import ReSE_UNet3D
from glioma_seg.training import DataManager, Trainer
from glioma_seg.utils.logging import get_logger
from glioma_seg.utils.seed import set_seed


def build_model(cfg: DictConfig) -> torch.nn.Module:
    return ReSE_UNet3D(
        input_channels=cfg.model.input_channels,
        output_channels=cfg.model.output_channels,
        dropout_rate=cfg.model.dropout_rate,
        dropout_scaling=cfg.model.dropout_scaling,
        gradient_checkpointing=cfg.model.gradient_checkpointing,
    )


def build_loss(cfg: DictConfig) -> torch.nn.Module:
    loss_cfg = cfg.training.loss
    return DiceFocalLoss(
        include_background=loss_cfg.include_background,
        to_onehot_y=loss_cfg.to_onehot_y,
        softmax=loss_cfg.softmax,
        lambda_dice=loss_cfg.lambda_dice,
        lambda_focal=loss_cfg.lambda_focal,
        gamma=loss_cfg.focal_gamma,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log = get_logger()
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # --- Data ---------------------------------------------------------------
    augmenter = Augmenter(
        flip_prob=cfg.training.augmentation.flip_prob,
        rotate_prob=cfg.training.augmentation.rotate_prob,
        affined_prob=cfg.training.augmentation.affined_prob,
    )
    dm = DataManager(
        processed_dir=cfg.data.processed_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        prefetch_factor=cfg.training.prefetch_factor,
        transform=augmenter.compose(),
    )
    train_loader = dm.get_train_loader()
    val_loader = dm.get_val_loader()

    # --- Model / loss / optim ----------------------------------------------
    model = build_model(cfg)
    loss_fn = build_loss(cfg)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        patience=cfg.training.scheduler.patience,
        factor=cfg.training.scheduler.factor,
        min_lr=cfg.training.scheduler.min_lr,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_func=loss_fn,
        accumulation_steps=cfg.training.accumulation_steps,
    )
    evaluator = Evaluator(model=model, loss_func=loss_fn)

    # --- Loop --------------------------------------------------------------
    history = {"epoch": [], "training_loss": [], "validation_loss": [], "dice_score": []}
    best_dice = -1.0

    for epoch in range(1, cfg.training.epochs + 1):
        log.info("Epoch %d/%d", epoch, cfg.training.epochs)
        train_loss = trainer.train(train_loader)
        log.info("  train_loss=%.4f", train_loss)

        if epoch % cfg.training.logging.val_interval == 0:
            val_loss, dice, _hd95 = evaluator.evaluate(val_loader, average_metric=True)
            scheduler.step(val_loss)
            dice_val = float(dice.item()) if hasattr(dice, "item") else float(dice)
            log.info("  val_loss=%.4f  val_dice=%.4f", val_loss, dice_val)

            history["epoch"].append(epoch)
            history["training_loss"].append(train_loss)
            history["validation_loss"].append(val_loss)
            history["dice_score"].append(dice_val * 100.0)

            if cfg.training.logging.save_best and dice_val > best_dice:
                best_dice = dice_val
                ckpt_path = os.path.join(cfg.output_dir, "best.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_dice": dice_val,
                    },
                    ckpt_path,
                )
                log.info("  ↑ new best (Dice=%.4f) — saved to %s", dice_val, ckpt_path)

    # Always save the final checkpoint + training history.
    torch.save(model.state_dict(), os.path.join(cfg.output_dir, "last.pt"))
    torch.save(history, os.path.join(cfg.output_dir, "history.pt"))
    log.info("Training complete. Artefacts in %s", cfg.output_dir)


if __name__ == "__main__":
    main()
