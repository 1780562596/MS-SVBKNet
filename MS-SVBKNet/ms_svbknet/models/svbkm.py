from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import initialize_weights


def _motion_kernel(size: int, angle: float, length: float) -> torch.Tensor:
    """Anti-aliased line kernel used only to initialize the learnable dictionary."""
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    theta = math.radians(angle)
    along = xx * math.cos(theta) + yy * math.sin(theta)
    across = -xx * math.sin(theta) + yy * math.cos(theta)
    kernel = torch.exp(-0.5 * (across / 0.55).square())
    kernel = kernel * torch.sigmoid((length / 2 - along.abs()) * 4.0)
    return kernel / kernel.sum().clamp_min(1e-8)


class SVBKM(nn.Module):
    """Predict a normalized, position-dependent blur-kernel field.

    ``learned`` and ``fixed`` use the paper's basis-kernel decomposition.  The
    latter freezes the deterministic motion-kernel bank for diagnostic studies.
    ``dense`` predicts all K^2 coefficients directly and is retained as an
    explicit diagnostic mode.
    """

    def __init__(
        self,
        in_channels: int = 64,
        num_kernels: int = 16,
        kernel_size: int = 15,
        dictionary_mode: Optional[str] = None,
        use_dictionary: Optional[bool] = None,
    ):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        if dictionary_mode is None:
            dictionary_mode = "learned" if use_dictionary is not False else "dense"
        if dictionary_mode not in {"learned", "fixed", "dense"}:
            raise ValueError(f"Unsupported dictionary_mode: {dictionary_mode}")
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.dictionary_mode = dictionary_mode
        out_channels = num_kernels if dictionary_mode != "dense" else kernel_size * kernel_size
        self.weight_predictor = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 1),
        )
        self.weight_predictor.apply(initialize_weights)
        if dictionary_mode != "dense":
            lengths = torch.linspace(1.0, float(kernel_size - 1), num_kernels)
            kernels = [_motion_kernel(kernel_size, 180.0 * i / num_kernels, float(lengths[i])) for i in range(num_kernels)]
            probabilities = torch.stack(kernels).unsqueeze(1).clamp_min(1e-6)
            if dictionary_mode == "learned":
                self.basis_logits = nn.Parameter(probabilities.log())
                self.register_buffer("fixed_basis", None)
            else:
                self.register_parameter("basis_logits", None)
                self.register_buffer("fixed_basis", probabilities / probabilities.sum((2, 3), keepdim=True))
        else:
            self.register_parameter("basis_logits", None)
            self.register_buffer("fixed_basis", None)

    def normalized_basis(self) -> Optional[torch.Tensor]:
        if self.basis_logits is not None:
            n = self.basis_logits.shape[0]
            return self.basis_logits.flatten(1).softmax(dim=1).view(n, 1, self.kernel_size, self.kernel_size)
        return self.fixed_basis

    def forward(self, degradation: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        weights = self.weight_predictor(degradation).softmax(dim=1)
        return weights, self.normalized_basis()


def basis_mixture_blur(image: torch.Tensor, weights: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Exact linear-basis implementation of K(x,y) spatial convolution.

    It avoids materializing BxCxK^2xHxW patches. By linearity,
    sum_i w_i(x,y) [k_i * I](x,y) equals the paper's local mixture kernel.
    """
    _, channels, h, w = image.shape
    n, _, kernel, _ = basis.shape
    if weights.shape[1] != n or weights.shape[-2:] != (h, w):
        raise ValueError("Kernel weights and image have incompatible shapes")
    pad = kernel // 2
    mode = "reflect" if h > pad and w > pad else "replicate"
    padded = F.pad(image, (pad, pad, pad, pad), mode=mode)
    # One grouped convolution computes every (channel, basis) response.  Weight
    # ordering is C groups, N filters per group, then reshaped back to B,N,C,H,W.
    grouped_bank = basis.to(dtype=image.dtype).repeat(channels, 1, 1, 1)
    responses = F.conv2d(padded, grouped_bank, groups=channels)
    stacked = responses.view(image.shape[0], channels, n, h, w).permute(0, 2, 1, 3, 4)
    return (stacked * weights.unsqueeze(2)).sum(dim=1)


def direct_spatial_blur(image: torch.Tensor, kernel_weights: torch.Tensor, kernel_size: int, chunk_rows: int = 32) -> torch.Tensor:
    """Memory-bounded direct per-pixel kernel application for diagnostic use."""
    b, c, h, w = image.shape
    k2 = kernel_size * kernel_size
    if kernel_weights.shape != (b, k2, h, w):
        raise ValueError("Direct kernel tensor must be B x K^2 x H x W")
    pad = kernel_size // 2
    mode = "reflect" if h > pad and w > pad else "replicate"
    padded = F.pad(image, (pad, pad, pad, pad), mode=mode)
    rows = []
    for y0 in range(0, h, chunk_rows):
        y1 = min(y0 + chunk_rows, h)
        source = padded[..., y0 : y1 + kernel_size - 1, :]
        patches = F.unfold(source, kernel_size=kernel_size)
        patches = patches.view(b, c, k2, y1 - y0, w)
        local = kernel_weights[..., y0:y1, :].unsqueeze(1)
        rows.append((patches * local).sum(dim=2))
    return torch.cat(rows, dim=-2)


def spatially_varying_blur(
    image: torch.Tensor,
    weights: torch.Tensor,
    basis: Optional[torch.Tensor],
    kernel_size: int,
) -> torch.Tensor:
    if basis is not None:
        return basis_mixture_blur(image, weights, basis)
    return direct_spatial_blur(image, weights, kernel_size)
