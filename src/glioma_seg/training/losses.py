"""Loss construction from the ``training.loss`` config block."""

from monai.losses import DiceFocalLoss, GeneralizedDiceFocalLoss
from torch import nn

LOSSES = ("generalized_dice_focal", "dice_focal")


def build_loss(loss_cfg) -> nn.Module:
    """Build the segmentation loss described by ``loss_cfg``.

    ``loss_cfg.name`` selects the MONAI loss:

    - ``generalized_dice_focal``: ``GeneralizedDiceFocalLoss`` (Generalized
      Dice + Focal); ``lambda_dice`` is passed as ``lambda_gdl``.
    - ``dice_focal``: ``DiceFocalLoss`` (Dice + Focal).
    """
    common = {
        "include_background": loss_cfg.include_background,
        "to_onehot_y": loss_cfg.to_onehot_y,
        "softmax": loss_cfg.softmax,
        "gamma": loss_cfg.focal_gamma,
        "lambda_focal": loss_cfg.lambda_focal,
    }
    if loss_cfg.name == "generalized_dice_focal":
        return GeneralizedDiceFocalLoss(lambda_gdl=loss_cfg.lambda_dice, **common)
    if loss_cfg.name == "dice_focal":
        return DiceFocalLoss(lambda_dice=loss_cfg.lambda_dice, **common)
    raise ValueError(f"Unknown loss {loss_cfg.name!r}; expected one of {LOSSES}")
