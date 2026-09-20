from __future__ import annotations

import torch
from torch import nn

from .blocks import initialize_weights


class ImprovedPatchDiscriminator(nn.Module):
    """Six-layer PatchGAN, no BatchNorm, with optional global average score."""

    def __init__(self, global_pool: bool = False):
        super().__init__()
        channels = [3, 64, 64, 128, 256, 512, 1]
        strides = [2, 2, 2, 2, 1, 1]
        layers = []
        for index, stride in enumerate(strides):
            layers.append(nn.Conv2d(channels[index], channels[index + 1], 4, stride=stride, padding=1))
            if index < len(strides) - 1:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.features = nn.Sequential(*layers)
        self.global_pool = global_pool
        self.apply(initialize_weights)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        patches = self.features(image)
        return patches.mean(dim=(-2, -1), keepdim=True) if self.global_pool else patches


def critic_mean(discriminator: nn.Module, image: torch.Tensor) -> torch.Tensor:
    return discriminator(image).mean()


def gradient_penalty(discriminator: nn.Module, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    batch = real.shape[0]
    alpha = torch.rand(batch, 1, 1, 1, device=real.device, dtype=real.dtype)
    mixed = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    score = discriminator(mixed).flatten(1).mean(dim=1)
    gradients = torch.autograd.grad(
        outputs=score,
        inputs=mixed,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.flatten(1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()
