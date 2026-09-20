from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import conv_relu, initialize_weights


class MDRM(nn.Module):
    """Shallow, middle and deep degradation branches aligned at full resolution."""

    def __init__(self, out_channels: int = 64):
        super().__init__()
        self.shallow = nn.Sequential(conv_relu(3, 32), conv_relu(32, 32), conv_relu(32, 32))
        self.middle = nn.Sequential(conv_relu(3, 32, 5, 2), conv_relu(32, 64), conv_relu(64, 64))
        self.deep = nn.Sequential(
            conv_relu(3, 32, 5, 2),
            conv_relu(32, 64, 3, 2),
            conv_relu(64, 128, 3, 1, dilation=2),
        )
        self.align1 = nn.Conv2d(32, 32, 1)
        self.align2 = nn.Conv2d(64, 32, 1)
        self.align3 = nn.Conv2d(128, 32, 1)
        self.fuse = nn.Sequential(nn.Conv2d(96, out_channels, 1), nn.ReLU(inplace=True))
        self.apply(initialize_weights)

    def forward(self, blurred: torch.Tensor) -> torch.Tensor:
        target = blurred.shape[-2:]
        f1 = self.align1(self.shallow(blurred))
        f2 = F.interpolate(self.align2(self.middle(blurred)), target, mode="bilinear", align_corners=False)
        f3 = F.interpolate(self.align3(self.deep(blurred)), target, mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([f1, f2, f3], dim=1))

