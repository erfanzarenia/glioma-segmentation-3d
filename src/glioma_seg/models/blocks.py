"""Building blocks for the 3D Residual-SE U-Net."""

import torch
import torch.nn as nn


class SE_Block(nn.Module):
    """Squeeze-and-Excitation (SE) block for 3D convolutional networks.

    Parameters
    ----------
    channels : int
        Number of input feature channels.
    reduction_factor : int
        Reduction factor for the bottleneck.

    Returns
    -------
    Tensor
        Recalibrated input features after applying channel-wise attention.
    """

    def __init__(self, channels, reduction_factor):
        super().__init__()

        # Squeeze
        self.squeeze = nn.AdaptiveAvgPool3d(1)  # Reduce the spatial dimensions

        # Excitation
        self.excitation1 = nn.Conv3d(  # First 1x1x1 convolutional layer
            in_channels=channels,
            out_channels=channels // reduction_factor,  # Reduce channels
            kernel_size=1,
        )

        self.excitation2 = nn.Conv3d(  # Second 1x1x1 convolutional layer
            in_channels=channels // reduction_factor,
            out_channels=channels,  # Restore original number of channels
            kernel_size=1,
        )

        # Introduce non-linearity between 1x1x1 convolutional layers via Rectified Linear Unit
        self.relu = nn.ReLU()

    def forward(self, input):
        sqz = self.squeeze(input)  # Apply squeeze
        exc_1 = self.excitation1(sqz)  # Apply first 1x1x1 convolutional layer
        exc_1r = self.relu(exc_1)  # Apply Non-linearity via ReLU
        exc_2 = self.excitation2(exc_1r)  # Apply second 1x1x1 convolutional layer
        exc_factor = torch.sigmoid(exc_2)  # Apply gating via sigmoid to get scale factors in (0, 1)

        # Scale each channel by its learned excitation factor
        return input * exc_factor


class Residual_SE_Block3D(nn.Module):
    """A flexible 3D convolutional block with optional residual connection and integrated SE attention.

    Structure
    ---------
        Conv3D → InstanceNorm → ReLU →
        Conv3D → InstanceNorm → ReLU →
        Squeeze-and-Excitation →
        + Residual (optional)

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    output_channels : list or tuple of int
        ``[C1, C2]`` where:
            - ``C1`` is the number of output channels from the first conv layer.
            - ``C2`` is the final output channel size (after second conv layer and SE block).
    kernel_sizes : list of int
        Kernel sizes for the two convolution layers. Default ``[3, 3]``.
    use_residual : bool
        If True, adds a residual connection from input to output.

    Returns
    -------
    Tensor
        Shape ``[B, C2, D, H, W]`` after convolution, normalization, SE, and optional residual.
    """

    def __init__(self, input_channels, output_channels, kernel_sizes=[3, 3], use_residual=False):
        super().__init__()

        # Calculate padding
        padding1 = (kernel_sizes[0] - 1) // 2  # Padding for the first kernel
        padding2 = (kernel_sizes[1] - 1) // 2  # Padding for the second kernel

        # First convolutional layer
        self.conv1 = nn.Conv3d(
            in_channels=input_channels,
            out_channels=output_channels[0],
            kernel_size=kernel_sizes[0],
            padding=padding1,
        )

        # Second convolutional layer
        self.conv2 = nn.Conv3d(
            in_channels=output_channels[0],
            out_channels=output_channels[1],
            kernel_size=kernel_sizes[1],
            padding=padding2,
        )

        # Activation and Normalization
        self.relu = nn.ReLU()
        self.inst_norm1 = nn.InstanceNorm3d(output_channels[0], affine=True)
        self.inst_norm2 = nn.InstanceNorm3d(output_channels[1], affine=True)

        # Residual connection
        self.use_residual = use_residual
        if use_residual:
            # Ensure input and output dimensions match
            if input_channels != output_channels[1]:
                self.residual = nn.Conv3d(input_channels, output_channels[1], kernel_size=1)
            else:
                self.residual = None  # Apply no transformation

        # Squeeze-and-Excitation Block
        self.se = SE_Block(output_channels[1], reduction_factor=output_channels[1] // 8)

    def forward(self, input):
        # First Convolution
        conv_1raw = self.conv1(input)
        conv_1norm = self.inst_norm1(conv_1raw)
        conv_1out = self.relu(conv_1norm)

        # Second Convolution
        conv_2raw = self.conv2(conv_1out)
        conv_2norm = self.inst_norm2(conv_2raw)
        conv_2out = self.relu(conv_2norm)

        # Apply Squeeze-and-Excitation
        conv_2se = self.se(conv_2out)

        # Add residual (if enabled)
        if self.use_residual:
            if self.residual:
                res = self.residual(input)  # Adjust channels if needed
            else:
                res = input  # Direct residual connection

            output = conv_2se + res  # Add SE output and residual

        # No residual path
        else:
            output = conv_2se

        return output
