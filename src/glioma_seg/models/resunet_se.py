"""3D Residual + Squeeze-and-Excitation U-Net architecture."""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from glioma_seg.models.blocks import Residual_SE_Block3D


class ReSE_UNet3D(nn.Module):
    """A custom 3D U-Net with residual + Squeeze-and-Excitation blocks.

    Features
    --------
    - Residual blocks for stable gradient flow.
    - Squeeze-and-Excitation channel attention.
    - Adaptive dropout scaling across depth.
    - Optional gradient checkpointing to save memory.

    Parameters
    ----------
    input_channels : int
        Number of input modalities.
    output_channels : int
        Number of segmentation classes.
    dropout_rate : float
        Base dropout rate for regularisation.
    dropout_scaling : bool
        If True, dropout rate increases with depth.
    gradient_checkpointing : bool
        If True, enables checkpointing for saving memory.
    """

    def __init__(
        self,
        input_channels: int = 2,
        output_channels: int = 3,
        dropout_rate: float = 0.0,
        dropout_scaling: bool = False,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()

        # Initialize gradient checkpointing
        self.gradient_checkpointing = gradient_checkpointing
        self.checkpoint_reentrant = False  # Needed for checkpoint in PyTorch 2.0+

        # Controls progressive dropout scaling
        self.dropout_scaling = dropout_scaling
        scaler = 1.25 if dropout_scaling else 1

        # Initialize dropout layers
        self.drop1 = nn.Dropout3d(p=dropout_rate)
        self.drop2 = nn.Dropout3d(p=dropout_rate * scaler)
        self.drop3 = nn.Dropout3d(p=dropout_rate * (scaler**2))
        self.drop_neck = nn.Dropout3d(p=dropout_rate * (scaler**3))

        # ===== Encoder Path =====
        self.encod1 = Residual_SE_Block3D(input_channels, [32, 64], kernel_sizes=[5, 3])
        self.pool1 = nn.MaxPool3d(2)

        self.encod2 = Residual_SE_Block3D(64, [64, 128], kernel_sizes=[3, 3])
        self.pool2 = nn.MaxPool3d(2)

        self.encod3 = Residual_SE_Block3D(128, [128, 256], kernel_sizes=[3, 3], use_residual=True)
        self.pool3 = nn.MaxPool3d(2)

        # ===== Bottleneck =====
        self.bottleneck = Residual_SE_Block3D(
            256, [256, 256], kernel_sizes=[3, 3], use_residual=True
        )

        # ===== Decoder Path =====
        self.upsamp3 = nn.ConvTranspose3d(256, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.decod3 = Residual_SE_Block3D(
            256 + 128, [128, 64], kernel_sizes=[3, 3], use_residual=True
        )

        self.upsamp2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.decod2 = Residual_SE_Block3D(128 + 32, [64, 32], kernel_sizes=[3, 3])

        self.upsamp1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.decod1 = Residual_SE_Block3D(64 + 16, [64, 32], kernel_sizes=[3, 5])

        # ===== Output =====
        self.outconv = nn.Conv3d(32, output_channels, kernel_size=1)

    def forward(self, input):
        # ===== Encoder Path =====
        encod_1 = self.encod1(input)
        encod_1dp = self.drop1(encod_1)

        if self.gradient_checkpointing:
            encod_2 = checkpoint(
                self.encod2,
                self.pool1(encod_1dp),
                use_reentrant=self.checkpoint_reentrant,
            )
        else:
            encod_2 = self.encod2(self.pool1(encod_1dp))
        encod_2dp = self.drop2(encod_2)

        if self.gradient_checkpointing:
            encod_3 = checkpoint(
                self.encod3,
                self.pool2(encod_2dp),
                use_reentrant=self.checkpoint_reentrant,
            )
        else:
            encod_3 = self.encod3(self.pool2(encod_2dp))
        encod_3dp = self.drop3(encod_3)

        # ===== Bottleneck =====
        if self.gradient_checkpointing:
            bl_neck = checkpoint(
                self.bottleneck,
                self.pool3(encod_3dp),
                use_reentrant=self.checkpoint_reentrant,
            )
        else:
            bl_neck = self.bottleneck(self.pool3(encod_3dp))
        bl_neck_dp = self.drop_neck(bl_neck)

        # ===== Decoder Path =====
        upsamp_3 = self.upsamp3(bl_neck_dp)
        if self.gradient_checkpointing:
            decod_3 = checkpoint(
                self.decod3,
                torch.cat([upsamp_3, encod_3], dim=1),
                use_reentrant=self.checkpoint_reentrant,
            )
        else:
            decod_3 = self.decod3(torch.cat([upsamp_3, encod_3], dim=1))
        decod_3dp = self.drop3(decod_3)

        upsamp_2 = self.upsamp2(decod_3dp)
        if self.gradient_checkpointing:
            decod_2 = checkpoint(
                self.decod2,
                torch.cat([upsamp_2, encod_2], dim=1),
                use_reentrant=self.checkpoint_reentrant,
            )
        else:
            decod_2 = self.decod2(torch.cat([upsamp_2, encod_2], dim=1))
        decod_2dp = self.drop2(decod_2)

        upsamp_1 = self.upsamp1(decod_2dp)
        decod_1 = self.decod1(torch.cat([upsamp_1, encod_1], dim=1))
        decod_1dp = self.drop1(decod_1)

        # ===== Output =====
        return self.outconv(decod_1dp)
