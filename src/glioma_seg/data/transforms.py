"""On-the-fly 3D augmentation for training, built on MONAI."""

from monai.transforms import (
    Compose,
    EnsureTyped,
    RandAffined,
    RandBiasFieldd,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
)


class Augmenter:
    """Builds a MONAI ``Compose`` of augmentations from tunable hyperparameters."""

    def __init__(
        self,
        flip_prob: float = 0.5,
        rotate_prob: float = 0.5,
        affined_prob: float = 0.3,
    ):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.affined_prob = affined_prob

    def compose(self):
        """Return a MONAI ``Compose`` object with the desired transforms."""
        return Compose(
            [
                # ------ Spatial Augmentations ------
                RandFlipd(keys=["image", "label"], spatial_axis=0, prob=self.flip_prob),
                RandFlipd(keys=["image", "label"], spatial_axis=1, prob=self.flip_prob),
                RandFlipd(keys=["image", "label"], spatial_axis=2, prob=self.flip_prob),
                RandRotate90d(keys=["image", "label"], prob=self.rotate_prob, spatial_axes=(0, 1)),
                RandRotate90d(keys=["image", "label"], prob=self.rotate_prob, spatial_axes=(1, 2)),
                RandRotate90d(keys=["image", "label"], prob=self.rotate_prob, spatial_axes=(0, 2)),
                RandAffined(
                    keys=["image", "label"],
                    prob=self.affined_prob,
                    rotate_range=(0.17, 0.17, 0.17),  # ~10° in radians
                    scale_range=(0.9, 1.1),  # ±10%
                    translate_range=(10, 10, 10),  # ±10 voxels
                ),
                # ------ Intensity Augmentations ------
                RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                RandGaussianNoised(keys="image", std=0.1, prob=0.2),
                RandBiasFieldd(keys="image", degree=2, coeff_range=(0.0, 0.1), prob=0.3),
                EnsureTyped(keys=["image", "label"]),
            ]
        )
