"""Builds PyTorch ``DataLoader``s for the BraTS 2021 splits."""

import os

from monai.transforms import Compose
from torch.utils.data import DataLoader

from glioma_seg.data.dataset import BraTSDataset


class DataManager:
    """Prepares ``DataLoader``s for BraTS 2021 with optional on-the-fly 3D augments."""

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

    def get_train_loader(self):
        """Return a ``DataLoader`` for the training set with augmentation."""
        train_data = os.path.join(self.processed_dir, "train", "data")
        train_label = os.path.join(self.processed_dir, "train", "labels")
        return DataLoader(
            BraTSDataset(train_data, train_label, transform=self.transform),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

    def get_val_loader(self):
        """Return a ``DataLoader`` for the validation set (no augmentation)."""
        val_data = os.path.join(self.processed_dir, "val", "data")
        val_label = os.path.join(self.processed_dir, "val", "labels")
        return DataLoader(
            BraTSDataset(val_data, val_label),
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

    def get_test_loader(self):
        """Return a ``DataLoader`` for the test set."""
        test_data = os.path.join(self.processed_dir, "test", "data")
        test_label = os.path.join(self.processed_dir, "test", "labels")
        return DataLoader(
            BraTSDataset(test_data, test_label),
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )
