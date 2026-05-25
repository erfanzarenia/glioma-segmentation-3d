"""Evaluation loop with MONAI Dice + HD95."""

import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.utils import one_hot
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Evaluator:
    """Validation / test evaluation using MONAI metrics.

    Tracks
    ------
    - Dice score (per-class or mean).
    - 95th-percentile Hausdorff distance (per-class or mean).
    - Mean loss across the dataset.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_func: torch.nn.Module,
    ):
        self.model = model.to(DEVICE)
        self.loss_fn = loss_func

    def evaluate(self, loader: torch.utils.data.DataLoader, average_metric: bool = True):
        """Evaluate the model on the given dataloader.

        Parameters
        ----------
        loader : DataLoader
            Evaluation set (val or test).
        average_metric : bool
            If True, compute mean across classes. If False, mean per class.

        Returns
        -------
        tuple
            ``(avg_loss, dice, hd95)``.
        """
        self.model.eval()
        total_loss = 0.0

        avrg = "mean" if average_metric else "mean_batch"

        dice_metric = DiceMetric(
            include_background=False,
            reduction=avrg,
            num_classes=3,
        )

        hd95_metric = HausdorffDistanceMetric(
            include_background=False,
            percentile=95,
            reduction=avrg,
        )

        with torch.no_grad():
            for x, y in tqdm(loader, desc="Evaluate"):
                x, y = x.to(DEVICE), y.to(DEVICE)

                out = self.model(x)
                loss = self.loss_fn(out, y)
                total_loss += loss.item()

                pred = out.argmax(dim=1, keepdim=True).cpu()
                target = y.cpu()

                pred_oh = one_hot(pred, num_classes=3)
                target_oh = one_hot(target, num_classes=3)

                dice_metric(y_pred=pred_oh, y=target_oh)
                hd95_metric(y_pred=pred_oh, y=target_oh)

        dice = dice_metric.aggregate()
        hd95 = hd95_metric.aggregate()

        dice_metric.reset()
        hd95_metric.reset()

        avg_loss = total_loss / len(loader)

        return avg_loss, dice, hd95
