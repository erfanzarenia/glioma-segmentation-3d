"""I/O helpers for the BraTS 2021 dataset."""

import os

import nibabel as nib
import numpy as np
import torch


class DataExtractor:
    """Load raw NIfTI volumes (FLAIR, T1ce, segmentation) for a subject.

    Assumes all subject directories contain files ending with:
        - ``flair.nii.gz``
        - ``t1ce.nii.gz``
        - ``seg.nii.gz``
    """

    def __init__(self, raw_dir):
        self.raw_dir = raw_dir

    def load(self, subject):
        """Load the FLAIR, T1ce, and segmentation volumes for a given subject.

        Parameters
        ----------
        subject : str
            Folder name of the subject to load.

        Returns
        -------
        tuple of np.ndarray
            ``(FLAIR, T1ce, segmentation)``, all as float32.
        """
        subj_path = os.path.join(self.raw_dir, subject)
        paths = {"flair": None, "t1ce": None, "seg": None}

        for fname in os.listdir(subj_path):
            if fname.endswith("flair.nii.gz"):
                paths["flair"] = os.path.join(subj_path, fname)
            elif fname.endswith("t1ce.nii.gz"):
                paths["t1ce"] = os.path.join(subj_path, fname)
            elif fname.endswith("seg.nii.gz"):
                paths["seg"] = os.path.join(subj_path, fname)

        if not all(paths.values()):
            missing = [k for k, v in paths.items() if v is None]
            raise ValueError(f"Missing {missing} for {subject}")

        return (
            nib.load(paths["flair"]).get_fdata(dtype=np.float32),
            nib.load(paths["t1ce"]).get_fdata(dtype=np.float32),
            nib.load(paths["seg"]).get_fdata(dtype=np.float32),
        )


class DataSaver:
    """Save preprocessed data and labels into train/val/test splits.

    The split is determined by the sample index and the total number of raw
    subjects (i.e., a sorted, deterministic split — not randomised).
    """

    def __init__(self, raw_dir, output_dir, split_ratio=(0.8, 0.1, 0.1)):
        self.output_dir = output_dir
        self.raw_dir = raw_dir
        self.split_ratio = split_ratio

        self.total_samples = len(
            [d for d in os.listdir(self.raw_dir) if os.path.isdir(os.path.join(self.raw_dir, d))]
        )

    def save(self, idx, data, label):
        """Save ``data`` and ``label`` as torch tensors in the appropriate split.

        Parameters
        ----------
        idx : int
            Sample index (used to determine which split it goes to).
        data : np.ndarray
            Image tensor, shape ``(H, W, D, C)``.
        label : np.ndarray
            Label tensor, shape ``(H, W, D)``.
        """
        for subset in ["train", "val", "test"]:
            os.makedirs(os.path.join(self.output_dir, subset, "data"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, subset, "labels"), exist_ok=True)

        t_end = int(self.total_samples * self.split_ratio[0])
        v_end = int(self.total_samples * sum(self.split_ratio[:2]))

        if idx < t_end:
            split = "train"
        elif idx < v_end:
            split = "val"
        else:
            split = "test"

        data_t = torch.from_numpy(data).permute(3, 0, 1, 2)  # (H, W, D, C) → (C, H, W, D)
        label_t = torch.from_numpy(label).unsqueeze(0)  # (H, W, D)    → (1, H, W, D)

        torch.save(data_t, os.path.join(self.output_dir, split, "data", f"{idx + 1}_data.pt"))
        torch.save(label_t, os.path.join(self.output_dir, split, "labels", f"{idx + 1}_label.pt"))


class Inspector:
    """Verify the shapes and dtypes of processed data and label tensors."""

    def __init__(self, processed_dir):
        self.processed_dir = processed_dir

    def inspect(self, subset, data_shape, data_type, labels_shape, labels_type):
        """Check a specific split for correct tensor shapes and dtypes.

        Returns
        -------
        tuple
            ``(inspected_data, inspected_labels, irregular_data, irregular_labels)``.
        """
        split_path = os.path.join(self.processed_dir, subset)
        data_dir = os.path.join(split_path, "data")
        labels_dir = os.path.join(split_path, "labels")

        inspected_data = 0
        irregular_data = []
        for data_file in os.listdir(data_dir):
            data = torch.load(os.path.join(data_dir, data_file))
            inspected_data += 1
            if data.shape != torch.Size(data_shape) or data.dtype != data_type:
                irregular_data.append(data)

        inspected_labels = 0
        irregular_labels = []
        for label_file in os.listdir(labels_dir):
            label = torch.load(os.path.join(labels_dir, label_file))
            inspected_labels += 1
            if label.shape != torch.Size(labels_shape) or label.dtype != labels_type:
                irregular_labels.append(label)

        return inspected_data, inspected_labels, irregular_data, irregular_labels
