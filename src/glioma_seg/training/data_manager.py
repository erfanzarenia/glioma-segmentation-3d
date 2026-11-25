"""Builds PyTorch ``DataLoader``s for the BraTS 2021 splits."""

import os

from monai.transforms import Compose
from torch.utils.data import DataLoader

from glioma_seg.data.dataset import BraTSDataset


class DataManager:
    """Prepares ``DataLoader``s for BraTS 2021 with optional on-the-fly 3D augments.

    ``batch_size`` applies to the training loader only; validation and test
    loaders always use a batch size of 1. ``transform`` is applied to the
    training set only.
    """

    def __init__(
        self,
        processed_dir: str,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        transform: Compose = None,
    ):
        self.processed_dir = processed_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.transform = transform

    def _loader(self, split: str, batch_size: int, shuffle: bool, transform=None) -> DataLoader:
        data_dir = os.path.join(self.processed_dir, split, "data")
        label_dir = os.path.join(self.processed_dir, split, "labels")
        return DataLoader(
            BraTSDataset(data_dir, label_dir, transform=transform),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # PyTorch rejects prefetch_factor when loading in the main process.
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def get_train_loader(self):
        """Return a shuffled ``DataLoader`` for the training set with augmentation."""
        return self._loader("train", self.batch_size, shuffle=True, transform=self.transform)

    def get_val_loader(self):
        """Return a ``DataLoader`` for the validation set (no augmentation)."""
        return self._loader("val", 1, shuffle=False)

    def get_test_loader(self):
        """Return a ``DataLoader`` for the test set (no augmentation)."""
        return self._loader("test", 1, shuffle=False)
