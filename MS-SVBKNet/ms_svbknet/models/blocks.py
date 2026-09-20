from __future__ import annotations

import torch
from torch import nn


class ImprovedResidualBlock(nn.Module):
    """Three 3x3 convolutions, hierarchical dropout, no normalization."""

    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.relu(self.conv1(x))
        residual = self.dropout(residual)
        residual = self.relu(self.conv2(residual))
        residual = self.conv3(residual)
        return self.relu(x + residual)


def conv_relu(in_channels: int, out_channels: int, kernel: int = 3, stride: int = 1, dilation: int = 1):
    padding = dilation * (kernel // 2)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding, dilation=dilation),
        nn.ReLU(inplace=True),
    )


def initialize_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

