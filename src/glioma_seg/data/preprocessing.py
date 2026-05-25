"""End-to-end preprocessing pipeline for BraTS 2021.

This module:
    - Computes global non-zero crop bounds and intensity statistics in one pass.
    - Crops every volume to that common bbox, z-score normalises each modality,
      refines labels, and saves preprocessed tensors split into train/val/test.
"""

import os

import numpy as np

from glioma_seg.data.io import DataExtractor, DataSaver


def calculate_crop_range(min_x, max_x, min_y, max_y, min_z, max_z):
    """Round the cropping bounds to the nearest multiple of 8.

    Needed because the U-Net pools spatial dims three times (factor of 8).
    """

    def _crop(a, b):
        size = b - a + 1
        rem = size % 8
        new_size = size - rem if rem <= 4 else size + (8 - rem)
        return (a, a + new_size)

    return _crop(min_x, max_x), _crop(min_y, max_y), _crop(min_z, max_z)


def volume_crop(volume, crop_x, crop_y, crop_z):
    """Crop a 3D volume to the provided bounds."""
    return volume[crop_x[0] : crop_x[1], crop_y[0] : crop_y[1], crop_z[0] : crop_z[1]]


def z_score(data, mean, std):
    """Z-score normalise voxel intensities."""
    return (data - mean) / std


def refine_label(label_data):
    """Convert BraTS label format ``{0, 1, 2, 4}`` to ``{0, 1, 1, 2}``.

    - ``0``: background (unchanged)
    - ``1``: merged non-enhancing + edema (original 1 and 2 collapsed)
    - ``2``: enhancing tumour (originally 4)
    """
    lbl = label_data.copy().astype(np.int64)
    lbl[np.isin(lbl, [1, 2])] = 1
    lbl[lbl == 4] = 2
    return lbl


class Preprocessor:
    """End-to-end preprocessing pipeline."""

    def __init__(self, raw_dir, processed_dir, split_ratio=(0.8, 0.1, 0.1)):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.split_ratio = split_ratio
        self.crop_bounds = None
        self.intensity_stats = None
        self.extractor = DataExtractor(raw_dir)
        self.saver = DataSaver(raw_dir, processed_dir, split_ratio)

    def compute_metadata(self):
        """Compute global non-zero crop bounds and intensity statistics."""
        mins = [np.inf, np.inf, np.inf]
        maxs = [-np.inf, -np.inf, -np.inf]

        sums = {"flair": 0.0, "t1c": 0.0}
        sqs = {"flair": 0.0, "t1c": 0.0}
        cnts = {"flair": 0, "t1c": 0}

        for subj in sorted(os.listdir(self.raw_dir)):
            flair, t1c, label = self.extractor.load(subj)

            for vol in (flair, t1c):
                coords = np.argwhere(vol != 0)
                for d in range(3):
                    mins[d] = min(mins[d], coords[:, d].min())
                    maxs[d] = max(maxs[d], coords[:, d].max())

            zcoords = np.where(np.any(label != 0, axis=(0, 1)))[0]
            mins[2] = min(mins[2], zcoords[0])
            maxs[2] = max(maxs[2], zcoords[-1])

            for key, vol in (("flair", flair), ("t1c", t1c)):
                mask = vol != 0
                vals = vol[mask]
                sums[key] += vals.sum()
                sqs[key] += np.square(vals).sum()
                cnts[key] += vals.size

        cx, cy, cz = calculate_crop_range(
            int(mins[0]),
            int(maxs[0]),
            int(mins[1]),
            int(maxs[1]),
            int(mins[2]),
            int(maxs[2]),
        )
        self.crop_bounds = {"x": cx, "y": cy, "z": cz}

        stats = {}
        for key in ("flair", "t1c"):
            mean = sums[key] / cnts[key]
            std = np.sqrt(sqs[key] / cnts[key] - mean**2)
            stats[f"{key}_mean"] = mean
            stats[f"{key}_std"] = std

        self.intensity_stats = stats
        return self.crop_bounds, self.intensity_stats

    def run(self, crop_bounds, stats):
        """Run the full preprocessing on all subjects using precomputed metadata."""
        for idx, subj in enumerate(sorted(os.listdir(self.raw_dir))):
            flair, t1c, label = self.extractor.load(subj)

            cf = volume_crop(flair, *crop_bounds.values())
            ct = volume_crop(t1c, *crop_bounds.values())
            cl = volume_crop(label, *crop_bounds.values())

            nf = z_score(cf, stats["flair_mean"], stats["flair_std"])
            nt = z_score(ct, stats["t1c_mean"], stats["t1c_std"])

            combined = np.stack((nf, nt), axis=-1)
            refined = refine_label(cl)

            self.saver.save(idx, combined, refined)

        print(
            f"All {len(os.listdir(self.raw_dir))} FLAIR and T1c images along with "
            "their corresponding labels have been successfully processed and saved."
        )
