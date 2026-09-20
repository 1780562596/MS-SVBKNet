from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


def pixel_loss(restored: torch.Tensor, sharp: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(restored, sharp)


def frequency_loss(restored: torch.Tensor, sharp: torch.Tensor) -> torch.Tensor:
    # Complex-spectrum L1 distance corresponding to ||F(Ir)-F(Ic)|| in the paper.
    restored_fft = torch.fft.rfft2(restored.float(), norm="ortho")
    sharp_fft = torch.fft.rfft2(sharp.float(), norm="ortho")
    return torch.abs(restored_fft - sharp_fft).mean()


def credible_reblur_loss(
    reblurred: torch.Tensor,
    blurred: torch.Tensor,
    alpha: float = 8.0,
    detach_reliability: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    difference = (reblurred - blurred).abs()
    local_error = difference.mean(dim=1, keepdim=True)
    reliability = torch.exp(-alpha * local_error)
    # Stop-gradient prevents the trivial solution of increasing error to shrink R.
    loss_weight = reliability.detach() if detach_reliability else reliability
    return (loss_weight * difference).mean(), reliability, local_error


@dataclass
class DynamicCredWeight:
    initial: float = 0.1
    maximum: float = 3.0
    momentum: float = 0.98
    reference: float | None = None
    ema: float | None = None

    def update(self, value: torch.Tensor) -> float:
        current = float(value.detach().float().item())
        if self.reference is None:
            self.reference = max(current, 1e-8)
            self.ema = self.reference
            return self.initial
        self.ema = self.momentum * float(self.ema) + (1.0 - self.momentum) * current
        return min(self.maximum, max(self.initial, self.initial * self.reference / max(self.ema, 1e-8)))

    def state_dict(self) -> dict:
        return {"reference": self.reference, "ema": self.ema}

    def load_state_dict(self, state: dict | None) -> None:
        if state:
            self.reference = state.get("reference")
            self.ema = state.get("ema")


def generator_adversarial_loss(discriminator, restored: torch.Tensor, mode: str = "wgan_gp") -> torch.Tensor:
    scores = discriminator(restored)
    if mode in {"wgan_gp", "hinge"}:
        # Direct implementation of L_adv = -E[D(I_r)] in the manuscript.
        return -scores.mean()
    if mode == "lsgan":
        return 0.5 * (scores - 1.0).square().mean()
    raise ValueError(f"Unsupported GAN mode: {mode}")


def discriminator_adversarial_loss(
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor,
    mode: str = "wgan_gp",
) -> torch.Tensor:
    if mode == "wgan_gp":
        return fake_scores.mean() - real_scores.mean()
    if mode == "hinge":
        return F.relu(1.0 - real_scores).mean() + F.relu(1.0 + fake_scores).mean()
    if mode == "lsgan":
        return 0.5 * ((real_scores - 1.0).square().mean() + fake_scores.square().mean())
    raise ValueError(f"Unsupported GAN mode: {mode}")
