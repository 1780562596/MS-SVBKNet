from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import ImprovedResidualBlock, initialize_weights


class Generator(nn.Module):
    """Johnson-style encoder-decoder with ten improved residual blocks."""

    def __init__(self, base_channels: int = 64, degradation_channels: int = 64, fuse_degradation: bool = True):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(3, c, 7), nn.ReLU(inplace=True))
        self.down1 = nn.Sequential(nn.Conv2d(c, c * 2, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.down2 = nn.Sequential(nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.fuse_degradation = fuse_degradation
        self.degradation_projection = nn.Conv2d(degradation_channels, c * 4, 1) if fuse_degradation else None
        dropouts = [0.0] * 3 + [0.15] * 4 + [0.25] * 3
        self.residuals = nn.Sequential(*[ImprovedResidualBlock(c * 4, rate) for rate in dropouts])
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c * 4, c * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c * 2, c, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(c, 3, 7))
        self.apply(initialize_weights)

    def forward(self, blurred: torch.Tensor, degradation: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = self.down2(self.down1(self.stem(blurred)))
        if self.fuse_degradation and degradation is not None:
            aligned = F.interpolate(degradation, size=features.shape[-2:], mode="bilinear", align_corners=False)
            features = features + self.degradation_projection(aligned)
        residual = self.head(self.up2(self.up1(self.residuals(features))))
        # Global residual learning from the paper's generator diagram.
        return torch.tanh(residual + blurred)

