from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from ms_svbknet.config import (
    architecture_fingerprint,
    config_fingerprint,
    load_config,
    to_plain,
    verify_checkpoint_architecture,
    verify_checkpoint_training,
)
from ms_svbknet.data import audit_pair_splits, make_loader, validate_expected_scene_counts
from ms_svbknet.engine import Trainer, record_epoch, validate
from ms_svbknet.models import ImprovedPatchDiscriminator, MSSVBKNet
from ms_svbknet.utils import environment_manifest, load_checkpoint, seed_everything, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Train MS-SVBKNet")
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--set", action="append", default=[], help="Override dotted key, e.g. train.batch_size=2")
    parser.add_argument("--resume", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg.experiment.seed, cfg.experiment.deterministic)
    device = torch.device(cfg.experiment.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg.experiment.output_dir) / cfg.experiment.name / f"seed_{cfg.experiment.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_plain(cfg), handle, sort_keys=False, allow_unicode=True)
    train_loader = make_loader(cfg, "train", True, cfg.experiment.seed)
    val_loader = make_loader(cfg, "val", False, cfg.experiment.seed)
    data_audit = audit_pair_splits(
        {"train": train_loader.dataset.pairs, "val": val_loader.dataset.pairs},
        blur_variant=str(cfg.data.blur_variant),
        check_scene_leakage=bool(getattr(cfg.data, "enforce_scene_disjoint", True)),
    )
    validate_expected_scene_counts(data_audit, cfg.data)
    expected_trainval = int(getattr(cfg.data, "expected_trainval_pairs", 0))
    actual_trainval = len(train_loader.dataset) + len(val_loader.dataset)
    if expected_trainval and actual_trainval != expected_trainval:
        raise RuntimeError(
            f"GoPro train+val count mismatch: expected {expected_trainval}, found {actual_trainval}. "
            "Check data.blur_variant and the split CSVs."
        )
    expected_scenes = int(getattr(cfg.data, "expected_trainval_scenes", 0))
    scene_counts = data_audit.get("scene_counts", {})
    actual_scenes = int(scene_counts.get("train", 0)) + int(scene_counts.get("val", 0))
    if expected_scenes and actual_scenes != expected_scenes:
        raise RuntimeError(
            f"GoPro train+val scene count mismatch: expected {expected_scenes}, found {actual_scenes}."
        )
    write_json(data_audit, output_dir / "data_audit.json")
    write_json(
        {
            "config_sha256": config_fingerprint(cfg),
            "architecture_sha256": architecture_fingerprint(cfg),
            "data_audit": data_audit,
            **environment_manifest(cfg),
        },
        output_dir / "experiment_manifest.json",
    )

    model = MSSVBKNet(cfg).to(device)
    discriminator = ImprovedPatchDiscriminator(cfg.model.discriminator_global_pool).to(device)
    trainer = Trainer(cfg, model, discriminator, device, output_dir)
    start_epoch, best_psnr = 1, float("-inf")
    if args.resume:
        checkpoint = load_checkpoint(args.resume, map_location=device)
        verify_checkpoint_architecture(checkpoint, cfg)
        verify_checkpoint_training(checkpoint, cfg)
        saved_hash = checkpoint.get("config_sha256")
        current_hash = config_fingerprint(cfg)
        if saved_hash and saved_hash != current_hash:
            print(
                "info: full config differs only in fields permitted by the exact-resume "
                f"signature (checkpoint={saved_hash}, current={current_hash})"
            )
        trainer.restore(checkpoint, train_loader)
        start_epoch = checkpoint["epoch"] + 1
        best_psnr = checkpoint.get("best_psnr", best_psnr)

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        train_values = trainer.train_epoch(train_loader, epoch)
        val_values = validate(cfg, model, val_loader, device)
        record_epoch(output_dir, epoch, train_values, val_values)
        is_best = val_values["psnr"] > best_psnr
        if is_best:
            best_psnr = val_values["psnr"]
        trainer.checkpoint(epoch, best_psnr, "last.pt", train_loader)
        if is_best:
            trainer.checkpoint(epoch, best_psnr, "best.pt", train_loader)
        print(f"epoch={epoch} val={val_values} best_psnr={best_psnr:.4f}")


if __name__ == "__main__":
    main()
