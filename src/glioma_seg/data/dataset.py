"""PyTorch ``Dataset`` for preprocessed BraTS 2021 tensors."""

import os

import torch
from torch.utils.data import Dataset


class BraTSDataset(Dataset):
    """Loads preprocessed BraTS 2021 tensors saved as ``.pt`` files.

    Expects ``data_dir`` and ``label_dir`` to contain parallel-named tensors.
    Applies a MONAI-style ``Compose`` transform on-the-fly if provided.
    """

    def __init__(self, data_dir, label_dir, transform=None):
        self.data_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)])
        self.label_paths = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data = torch.load(self.data_paths[idx])
        label = torch.load(self.label_paths[idx])

        sample = {"image": data, "label": label}

        if self.transform:
            sample = self.transform(sample)

        return sample["image"], sample["label"]
