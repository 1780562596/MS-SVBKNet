from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from tqdm import tqdm

from .config import architecture_fingerprint, config_fingerprint, to_plain, training_fingerprint
from .losses import (
    DynamicCredWeight,
    credible_reblur_loss,
    discriminator_adversarial_loss,
    frequency_loss,
    generator_adversarial_loss,
    pixel_loss,
)
from .metrics import MetricBundle
from .models.discriminator import gradient_penalty
from .utils import (
    append_csv,
    atomic_torch_save,
    capture_rng_state,
    restore_rng_state,
    save_triplet,
    tiled_restore,
)


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


class Trainer:
    def __init__(self, cfg, model: nn.Module, discriminator: nn.Module, device: torch.device, output_dir: Path):
        self.cfg, self.model, self.discriminator = cfg, model, discriminator
        self.device, self.output_dir = device, output_dir
        self.optimizer_g = torch.optim.Adam(
            model.generator_parameters(), lr=cfg.train.lr_generator, betas=tuple(cfg.train.betas)
        )
        blur_params = list(model.blur_parameters())
        self.optimizer_b = torch.optim.Adam(
            blur_params, lr=cfg.train.lr_blur, betas=tuple(cfg.train.betas)
        ) if blur_params else None
        self.optimizer_d = torch.optim.Adam(
            discriminator.parameters(), lr=cfg.train.lr_discriminator, betas=tuple(cfg.train.betas)
        )
        self.scaler = _make_grad_scaler(enabled=cfg.train.amp and device.type == "cuda")
        self.cred_scheduler = DynamicCredWeight(
            cfg.loss.cred_initial, cfg.loss.cred_maximum, cfg.loss.cred_momentum
        )
        self.gan_mode = str(getattr(cfg.loss, "gan_mode", "wgan_gp"))
        self.global_step = 0

    def _zero_generator(self):
        self.optimizer_g.zero_grad(set_to_none=True)
        if self.optimizer_b:
            self.optimizer_b.zero_grad(set_to_none=True)

    def train_epoch(self, loader, epoch: int) -> Dict[str, float]:
        self.model.train()
        self.discriminator.train()
        if hasattr(loader.dataset, "set_epoch"):
            loader.dataset.set_epoch(epoch)
        totals = defaultdict(float)
        progress = tqdm(loader, desc=f"train {epoch:03d}")
        steps = 0
        for batch in progress:
            blurred = batch["blur"].to(self.device, non_blocking=True)
            sharp = batch["sharp"].to(self.device, non_blocking=True)

            # Critic update in FP32 for a stable gradient penalty.
            with torch.no_grad():
                fake_for_d = self.model(blurred)["restored"]
            self.optimizer_d.zero_grad(set_to_none=True)
            real_scores = self.discriminator(sharp.float())
            fake_scores = self.discriminator(fake_for_d.detach().float())
            loss_d = discriminator_adversarial_loss(real_scores, fake_scores, self.gan_mode)
            gp = torch.zeros((), device=self.device)
            if self.gan_mode == "wgan_gp":
                gp = gradient_penalty(self.discriminator, sharp.float(), fake_for_d.detach().float())
                loss_d = loss_d + self.cfg.loss.gradient_penalty * gp
            loss_d.backward()
            self.optimizer_d.step()

            for parameter in self.discriminator.parameters():
                parameter.requires_grad_(False)
            self._zero_generator()
            with torch.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                outputs = self.model(blurred)
                restored = outputs["restored"]
                loss_pix = pixel_loss(restored, sharp)
                loss_freq = frequency_loss(restored, sharp)
                loss_adv = generator_adversarial_loss(self.discriminator, restored, self.gan_mode)
                loss_cred = torch.zeros((), device=self.device)
                cred_weight = 0.0
                if self.model.use_svbkm:
                    reblurred = self.model.reblur(restored, outputs)
                    if self.model.use_crcm:
                        loss_cred, reliability, _ = credible_reblur_loss(
                            reblurred,
                            blurred,
                            alpha=self.cfg.loss.reliability_alpha,
                            detach_reliability=self.cfg.loss.detach_reliability,
                        )
                        cred_weight = self.cred_scheduler.update(loss_cred)
                    else:
                        # Optional diagnostic mode: ordinary unweighted reblur consistency.
                        loss_cred = torch.nn.functional.l1_loss(reblurred, blurred)
                        cred_weight = self.cfg.loss.lambda_reblur
                loss_g = (
                    loss_pix
                    + self.cfg.loss.lambda_frequency * loss_freq
                    + self.cfg.loss.lambda_adversarial * loss_adv
                    + cred_weight * loss_cred
                )
            self.scaler.scale(loss_g).backward()
            self.scaler.unscale_(self.optimizer_g)
            if self.optimizer_b:
                self.scaler.unscale_(self.optimizer_b)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            self.scaler.step(self.optimizer_g)
            if self.optimizer_b:
                self.scaler.step(self.optimizer_b)
            self.scaler.update()
            for parameter in self.discriminator.parameters():
                parameter.requires_grad_(True)

            values = {
                "loss_g": float(loss_g.detach()), "loss_d": float(loss_d.detach()),
                "gradient_penalty": float(gp.detach()),
                "pixel": float(loss_pix.detach()), "frequency": float(loss_freq.detach()),
                "adversarial": float(loss_adv.detach()), "credible": float(loss_cred.detach()),
                "lambda_cred": cred_weight,
            }
            for key, value in values.items():
                totals[key] += value
            self.global_step += 1
            steps += 1
            preview_interval = int(getattr(self.cfg.train, "preview_interval", 0))
            if preview_interval > 0 and self.global_step % preview_interval == 0:
                save_triplet(
                    blurred.add(1).mul(0.5), restored.add(1).mul(0.5), sharp.add(1).mul(0.5),
                    self.output_dir / "previews" / f"step_{self.global_step:08d}.png",
                )
            progress.set_postfix(g=f"{values['loss_g']:.3f}", d=f"{values['loss_d']:.3f}")
            max_steps = int(getattr(self.cfg.train, "max_steps_per_epoch", 0))
            if max_steps > 0 and steps >= max_steps:
                break
        count = max(steps, 1)
        return {key: value / count for key, value in totals.items()}

    def checkpoint(self, epoch: int, best_psnr: float, name: str, train_loader=None) -> None:
        loader_generator = getattr(train_loader, "repro_generator", None)
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_psnr": best_psnr,
            "model": self.model.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_b": self.optimizer_b.state_dict() if self.optimizer_b else None,
            "optimizer_d": self.optimizer_d.state_dict(),
            "scaler": self.scaler.state_dict(),
            "credible_weight": self.cred_scheduler.state_dict(),
            "rng_state": capture_rng_state(),
            "config": to_plain(self.cfg),
            "config_sha256": config_fingerprint(self.cfg),
            "architecture_sha256": architecture_fingerprint(self.cfg),
            "training_sha256": training_fingerprint(self.cfg),
            "train_loader_generator_state": (
                loader_generator.get_state() if loader_generator is not None else None
            ),
        }
        atomic_torch_save(state, self.output_dir / "checkpoints" / name)

    def restore(self, checkpoint: dict, train_loader=None) -> None:
        self.model.load_state_dict(checkpoint["model"])
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        self.optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        if self.optimizer_b and checkpoint.get("optimizer_b"):
            self.optimizer_b.load_state_dict(checkpoint["optimizer_b"])
        self.optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        if checkpoint.get("scaler"):
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.cred_scheduler.load_state_dict(checkpoint.get("credible_weight"))
        self.global_step = int(checkpoint.get("global_step", 0))
        restore_rng_state(checkpoint.get("rng_state"))
        loader_state = checkpoint.get("train_loader_generator_state")
        loader_generator = getattr(train_loader, "repro_generator", None)
        if loader_state is not None:
            if loader_generator is None:
                raise RuntimeError("Checkpoint contains DataLoader RNG state but loader exposes no generator")
            loader_generator.set_state(loader_state)


@torch.no_grad()
def validate(cfg, model: nn.Module, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    metrics = MetricBundle(
        device,
        use_lpips=bool(getattr(cfg.train, "val_lpips", False)),
        color_space=getattr(cfg.eval, "color_space", "rgb"),
        metric_protocol=getattr(cfg.eval, "metric_protocol", "float_rgb"),
        lpips_backend=getattr(cfg.eval, "lpips_backend", "official_vgg16"),
    )
    totals = defaultdict(float)
    count = 0
    for batch in tqdm(loader, desc="validate", leave=False):
        blurred = batch["blur"].to(device, non_blocking=True)
        sharp = batch["sharp"].to(device, non_blocking=True)
        restored = tiled_restore(model, blurred, cfg.eval.tile, cfg.eval.overlap)
        values = metrics(restored, sharp, cfg.eval.crop_border)
        for key, value in values.items():
            totals[key] += value
        count += 1
        max_images = int(getattr(cfg.eval, "max_images", 0))
        if max_images > 0 and count >= max_images:
            break
    return {key: value / max(count, 1) for key, value in totals.items()}


def record_epoch(output_dir: Path, epoch: int, train_values: Dict[str, float], val_values: Dict[str, float]) -> None:
    row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_values.items()}, **{f"val_{k}": v for k, v in val_values.items()}}
    append_csv(row, output_dir / "history.csv")
