"""Training loop with AMP + gradient accumulation."""

import torch
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    """Single-epoch training loop with mixed precision and gradient accumulation."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        loss_func: torch.nn.Module,
        accumulation_steps: int,
    ):
        self.model = model.to(DEVICE)
        self.opt = optimizer
        self.sched = scheduler
        self.loss_fn = loss_func
        self.accum = accumulation_steps
        self.scaler = GradScaler()

    def train(self, loader: torch.utils.data.DataLoader):
        """Run one full training epoch and return the mean loss."""
        self.model.train()
        self.opt.zero_grad()
        total_loss = 0.0

        for step, (x, y) in enumerate(tqdm(loader, desc="Train")):
            x, y = x.to(DEVICE), y.to(DEVICE)

            with autocast(device_type=DEVICE.type):
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
