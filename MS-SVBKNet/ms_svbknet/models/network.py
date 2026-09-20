from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from .generator import Generator
from .mdrm import MDRM
from .svbkm import SVBKM, spatially_varying_blur


class MSSVBKNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.use_mdrm = bool(cfg.model.use_mdrm)
        self.use_svbkm = bool(cfg.model.use_svbkm)
        self.use_crcm = bool(cfg.model.use_crcm)
        self.kernel_size = int(cfg.model.kernel_size)
        self.dictionary_mode = getattr(
            cfg.model,
            "dictionary_mode",
            "learned" if bool(getattr(cfg.model, "use_dictionary", False)) else "dense",
        )
        degradation_channels = int(cfg.model.degradation_channels)
        self.mdrm = MDRM(degradation_channels) if self.use_mdrm else None
        self.generator = Generator(
            base_channels=int(cfg.model.base_channels),
            degradation_channels=degradation_channels,
            fuse_degradation=self.use_mdrm and bool(cfg.model.fuse_mdrm_to_generator),
        )
        self.svbkm = SVBKM(
            degradation_channels,
            num_kernels=int(cfg.model.num_kernels),
            kernel_size=self.kernel_size,
            dictionary_mode=self.dictionary_mode,
        ) if self.use_svbkm else None

    def forward(self, blurred: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        degradation = self.mdrm(blurred) if self.mdrm is not None else None
        restored = self.generator(blurred, degradation)
        weights = basis = None
        if self.svbkm is not None:
            if degradation is None:
                raise RuntimeError("SVBKM requires MDRM degradation features")
            weights, basis = self.svbkm(degradation)
        return {
            "restored": restored,
            "degradation": degradation,
            "kernel_weights": weights,
            "basis_kernels": basis,
        }

    def reblur(self, restored: torch.Tensor, outputs: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        if outputs["kernel_weights"] is None:
            raise RuntimeError("Reblurring requires SVBKM")
        return spatially_varying_blur(
            restored,
            outputs["kernel_weights"],
            outputs["basis_kernels"],
            self.kernel_size,
        )

    def generator_parameters(self):
        return self.generator.parameters()

    def blur_parameters(self):
        modules = [module for module in (self.mdrm, self.svbkm) if module is not None]
        for module in modules:
            yield from module.parameters()
