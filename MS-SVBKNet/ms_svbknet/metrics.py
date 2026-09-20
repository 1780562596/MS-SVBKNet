from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F


def to_unit(x: torch.Tensor) -> torch.Tensor:
    return x.add(1).mul(0.5).clamp(0, 1)


def rgb_to_matlab_y(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] != 3:
        raise ValueError("Y-channel evaluation requires three-channel RGB input")
    coefficients = x.new_tensor([65.481, 128.553, 24.966]).view(1, 3, 1, 1)
    return (x * coefficients).sum(dim=1, keepdim=True).add(16.0).div(255.0)


def quantize_8bit(x: torch.Tensor) -> torch.Tensor:
    """Match saving an RGB tensor as an 8-bit PNG and reading it back."""
    return x.mul(255.0).round().div(255.0)


def _prepare(
    restored: torch.Tensor,
    sharp: torch.Tensor,
    crop_border: int,
    color_space: str,
    metric_protocol: str,
):
    restored, sharp = to_unit(restored), to_unit(sharp)
    if metric_protocol == "gopro_matlab":
        restored, sharp = quantize_8bit(restored), quantize_8bit(sharp)
    elif metric_protocol != "float_rgb":
        raise ValueError(f"Unsupported metric protocol: {metric_protocol}")
    if crop_border:
        if min(restored.shape[-2:]) <= 2 * crop_border:
            raise ValueError("crop_border removes the complete image")
        restored = restored[..., crop_border:-crop_border, crop_border:-crop_border]
        sharp = sharp[..., crop_border:-crop_border, crop_border:-crop_border]
    if color_space == "y":
        restored, sharp = rgb_to_matlab_y(restored), rgb_to_matlab_y(sharp)
    elif color_space != "rgb":
        raise ValueError(f"Unsupported color space: {color_space}")
    return restored, sharp


def psnr(
    restored: torch.Tensor,
    sharp: torch.Tensor,
    crop_border: int = 0,
    color_space: str = "rgb",
    metric_protocol: str = "float_rgb",
) -> float:
    restored, sharp = _prepare(restored, sharp, crop_border, color_space, metric_protocol)
    mse = F.mse_loss(restored.double(), sharp.double()).item()
    return 100.0 if mse == 0 else -10.0 * math.log10(mse)


def _gaussian_window(size: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    window = (kernel[:, None] * kernel[None, :])[None, None]
    return window.expand(channels, 1, size, size).contiguous()


def ssim(
    restored: torch.Tensor,
    sharp: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    crop_border: int = 0,
    color_space: str = "rgb",
    metric_protocol: str = "float_rgb",
) -> float:
    x, y = _prepare(restored, sharp, crop_border, color_space, metric_protocol)
    x, y = x.double(), y.double()
    if min(x.shape[-2:]) < window_size:
        raise ValueError("image is too small for the requested SSIM window")
    channels = x.shape[1]
    window = _gaussian_window(window_size, sigma, channels, x.device, x.dtype)
    # Valid convolution is equivalent to skimage's Gaussian SSIM after its
    # border crop and follows DeepDeblur-PyTorch's public GoPro metric code.
    mu_x = F.conv2d(x, window, groups=channels)
    mu_y = F.conv2d(y, window, groups=channels)
    mu_x2, mu_y2, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    sigma_x = F.conv2d(x * x, window, groups=channels) - mu_x2
    sigma_y = F.conv2d(y * y, window, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, window, groups=channels) - mu_xy
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    )
    return float(score.mean().item())


class MetricBundle:
    def __init__(
        self,
        device: torch.device,
        use_lpips: bool = True,
        color_space: str = "rgb",
        metric_protocol: str = "float_rgb",
        lpips_backend: str = "official_vgg16",
    ):
        self.lpips_model: Optional[torch.nn.Module] = None
        self.color_space = color_space
        self.metric_protocol = metric_protocol
        self.lpips_backend = lpips_backend if use_lpips else "disabled"
        self.perceptual_metric = "disabled"
        self.perceptual_weights = "disabled"
        self.perceptual_backbone = "disabled"
        self.lpips_version = "disabled"
        self.lpips_calibrated = False
        if use_lpips:
            if lpips_backend != "official_vgg16":
                raise ValueError(f"Unsupported LPIPS backend: {lpips_backend}")
            try:
                import lpips
            except ImportError as exc:
                raise RuntimeError(
                    "Standard LPIPS-VGG16 requires the pinned 'lpips==0.1.4' package"
                ) from exc
            self.lpips_model = lpips.LPIPS(
                pretrained=True,
                net="vgg",
                version="0.1",
                lpips=True,
                pnet_rand=False,
                pnet_tune=False,
                eval_mode=True,
                verbose=False,
            ).to(device).eval()
            self.perceptual_metric = "official_lpips_vgg16_v0.1"
            self.perceptual_weights = "official_bapps_v0.1_vgg"
            self.perceptual_backbone = "vgg16"
            self.lpips_version = "0.1"
            self.lpips_calibrated = True
            for parameter in self.lpips_model.parameters():
                parameter.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, restored: torch.Tensor, sharp: torch.Tensor, crop_border: int = 0) -> Dict[str, float]:
        values = {
            "psnr": psnr(
                restored, sharp, crop_border, self.color_space, self.metric_protocol
            ),
            "ssim": ssim(
                restored,
                sharp,
                crop_border=crop_border,
                color_space=self.color_space,
                metric_protocol=self.metric_protocol,
            ),
        }
        if self.lpips_model is not None:
            perceptual_restored, perceptual_sharp = _prepare(
                restored, sharp, crop_border, "rgb", self.metric_protocol
            )
            perceptual_restored = perceptual_restored.mul(2).sub(1).float()
            perceptual_sharp = perceptual_sharp.mul(2).sub(1).float()
            values["lpips"] = float(
                self.lpips_model(perceptual_restored, perceptual_sharp).mean().item()
            )
        return values
