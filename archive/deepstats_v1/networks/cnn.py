"""CNN backbone for image inputs.

This module provides a CNN backbone for processing image inputs
in heterogeneous treatment effects estimation.

Examples
--------
>>> from deepstats.networks.cnn import CNNBackbone
>>> backbone = CNNBackbone(in_channels=3, image_size=32)
>>> x = torch.randn(16, 3, 32, 32)  # batch of 16 RGB images
>>> features = backbone(x)  # (16, output_dim)
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .base import BackboneNetwork
from .registry import ArchitectureRegistry


@ArchitectureRegistry.register("cnn")
class CNNBackbone(BackboneNetwork):
    """CNN backbone for image inputs.

    Processes image tensors and outputs a fixed-size feature vector.
    Uses a simple but effective architecture with Conv2d, BatchNorm, and ReLU.

    Parameters
    ----------
    in_channels : int, default=3
        Number of input channels (e.g., 3 for RGB, 1 for grayscale).
    image_size : int, default=32
        Height and width of input images (assumes square).
    hidden_channels : list[int], default=[32, 64, 128]
        Number of channels in each convolutional block.
    output_dim : int, default=64
        Dimension of output feature vector.
    kernel_size : int, default=3
        Kernel size for convolutions.
    dropout : float, default=0.0
        Dropout rate after global pooling.

    Attributes
    ----------
    input_dim : int
        Total input dimension (channels * height * width).
    output_dim : int
        Dimension of output features.

    Examples
    --------
    >>> backbone = CNNBackbone(in_channels=3, image_size=32)
    >>> x = torch.randn(8, 3, 32, 32)
    >>> out = backbone(x)
    >>> print(out.shape)  # (8, 64)
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        hidden_channels: Sequence[int] = (32, 64, 128),
        output_dim: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.0,
        input_dim: int | None = None,  # For registry compatibility
        **kwargs,
    ) -> None:
        # Compute input_dim for parent
        computed_input_dim = in_channels * image_size * image_size
        super().__init__(input_dim=computed_input_dim, hidden_dims=[output_dim], dropout=dropout)
        self.in_channels = in_channels
        self.image_size = image_size
        self._input_dim = computed_input_dim
        self._output_dim = output_dim
        hidden_channels = list(hidden_channels)

        # Build convolutional blocks
        layers = []
        current_channels = in_channels

        for out_channels in hidden_channels:
            layers.extend([
                nn.Conv2d(
                    current_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # Downsample by 2
            ])
            current_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Calculate size after conv layers
        # Each MaxPool2d reduces size by half
        final_size = image_size // (2 ** len(hidden_channels))
        self.flatten_dim = current_channels * final_size * final_size

        # Final layers: global pooling + linear
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_channels[-1], output_dim)

    @property
    def input_dim(self) -> int:
        """Total input dimension."""
        return self._input_dim

    @property
    def output_dim(self) -> int:
        """Output feature dimension."""
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, height, width).

        Returns
        -------
        torch.Tensor
            Feature tensor of shape (batch, output_dim).
        """
        # If input is flattened, reshape to image
        if x.dim() == 2:
            batch_size = x.shape[0]
            x = x.view(batch_size, self.in_channels, self.image_size, self.image_size)

        # Convolutional feature extraction
        x = self.conv_layers(x)

        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Dropout and final projection
        x = self.dropout(x)
        x = self.fc(x)

        return x


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection with projection if dimensions change
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection."""
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = nn.functional.relu(out, inplace=True)

        return out


@ArchitectureRegistry.register("resnet")
class ResNetBackbone(BackboneNetwork):
    """ResNet-style backbone for image inputs.

    Uses residual connections for better gradient flow.

    Parameters
    ----------
    in_channels : int, default=3
        Number of input channels.
    image_size : int, default=32
        Input image size (height = width).
    hidden_channels : list[int], default=[32, 64, 128]
        Channels in each residual block group.
    blocks_per_stage : int, default=2
        Number of residual blocks per stage.
    output_dim : int, default=64
        Output feature dimension.
    dropout : float, default=0.0
        Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        hidden_channels: Sequence[int] = (32, 64, 128),
        blocks_per_stage: int = 2,
        output_dim: int = 64,
        dropout: float = 0.0,
        input_dim: int | None = None,
        **kwargs,
    ) -> None:
        computed_input_dim = in_channels * image_size * image_size
        super().__init__(input_dim=computed_input_dim, hidden_dims=[output_dim], dropout=dropout)
        self.in_channels = in_channels
        self.image_size = image_size
        self._input_dim = computed_input_dim
        self._output_dim = output_dim
        hidden_channels = list(hidden_channels)

        # Initial conv
        self.conv1 = nn.Conv2d(in_channels, hidden_channels[0], 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels[0])

        # Build residual stages
        stages = []
        current_channels = hidden_channels[0]

        for i, out_channels in enumerate(hidden_channels):
            # First block may downsample
            stride = 2 if i > 0 else 1
            blocks = [ResidualBlock(current_channels, out_channels, stride=stride)]

            # Additional blocks
            for _ in range(blocks_per_stage - 1):
                blocks.append(ResidualBlock(out_channels, out_channels))

            stages.append(nn.Sequential(*blocks))
            current_channels = out_channels

        self.stages = nn.ModuleList(stages)

        # Global pooling and output
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_channels[-1], output_dim)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Reshape if flattened
        if x.dim() == 2:
            batch_size = x.shape[0]
            x = x.view(batch_size, self.in_channels, self.image_size, self.image_size)

        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x, inplace=True)

        # Residual stages
        for stage in self.stages:
            x = stage(x)

        # Global pooling and output
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
