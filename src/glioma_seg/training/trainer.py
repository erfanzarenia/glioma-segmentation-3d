"""Training loop with AMP + gradient accumulation."""

import torch
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from glioma_seg.utils.device import resolve_device


class Trainer:
    """Single-epoch training loop with mixed precision and gradient accumulation.

    Parameters
    ----------
    accumulation_steps : int
        Number of mini-batches whose gradients are summed before each
        optimizer step (effective batch = batch size x accumulation steps).
    device : str or torch.device, optional
        ``"auto"`` (default) uses CUDA when available, otherwise the CPU.
    amp : bool
        Mixed precision via ``torch.amp.autocast``. Loss scaling with
        ``GradScaler`` is only active on CUDA.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        loss_func: torch.nn.Module,
        accumulation_steps: int,
        device: str | torch.device | None = "auto",
        amp: bool = True,
    ):
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.opt = optimizer
        self.sched = scheduler
        self.loss_fn = loss_func
        self.accum = accumulation_steps
        self.amp = amp
        self.scaler = GradScaler(self.device.type, enabled=amp and self.device.type == "cuda")

    def train(self, loader: torch.utils.data.DataLoader):
        """Run one full training epoch and return the mean loss."""
        self.model.train()
        self.opt.zero_grad()
        total_loss = 0.0

        for step, (images, labels) in enumerate(tqdm(loader, desc="Train")):
            x, y = images.to(self.device), labels.to(self.device)

            with autocast(device_type=self.device.type, enabled=self.amp):
                out = self.model(x)
                loss = self.loss_fn(out, y)
                total_loss += loss.item()

            loss = loss / self.accum
            self.scaler.scale(loss).backward()

            if (step + 1) % self.accum == 0 or (step + 1) == len(loader):
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()

        return total_loss / len(loader)
