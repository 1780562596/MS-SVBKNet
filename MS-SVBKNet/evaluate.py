from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from ms_svbknet.config import (
    architecture_fingerprint,
    config_fingerprint,
    load_config,
    verify_checkpoint_architecture,
)
from ms_svbknet.data import audit_pair_splits, make_loader
from ms_svbknet.metrics import MetricBundle
from ms_svbknet.models import MSSVBKNet
from ms_svbknet.utils import (
    load_checkpoint,
    output_path_for_input,
    seed_everything,
    sha256_file,
    tensor_to_image,
    tiled_restore,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate on the complete paired test set")
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="results/evaluation")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--save-images", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg.experiment.seed, True)
    device = torch.device(cfg.experiment.device if torch.cuda.is_available() else "cpu")
    model = MSSVBKNet(cfg).to(device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    verify_checkpoint_architecture(checkpoint, cfg)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()
    loader = make_loader(cfg, "test", False, cfg.experiment.seed)
    data_audit = audit_pair_splits(
        {"test": loader.dataset.pairs}, blur_variant=str(cfg.data.blur_variant)
    )
    expected_test = int(getattr(cfg.data, "expected_test_pairs", 0))
    if expected_test and len(loader.dataset) != expected_test:
        raise RuntimeError(
            f"GoPro test count mismatch: expected {expected_test}, found {len(loader.dataset)}. "
            "Check data.blur_variant and data.test_index."
        )
    expected_test_scenes = int(getattr(cfg.data, "expected_test_scenes", 0))
    actual_test_scenes = int(data_audit.get("scene_counts", {}).get("test", 0))
    if expected_test_scenes and actual_test_scenes != expected_test_scenes:
        raise RuntimeError(
            f"GoPro test scene count mismatch: expected {expected_test_scenes}, "
            f"found {actual_test_scenes}."
        )
    metric_bundle = MetricBundle(
        device,
        bool(cfg.eval.lpips),
        getattr(cfg.eval, "color_space", "rgb"),
        getattr(cfg.eval, "metric_protocol", "float_rgb"),
        getattr(cfg.eval, "lpips_backend", "official_vgg16"),
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows, totals = [], defaultdict(float)
    with torch.no_grad():
        for batch in tqdm(loader, desc="test"):
            blurred, sharp = batch["blur"].to(device), batch["sharp"].to(device)
            restored = tiled_restore(model, blurred, cfg.eval.tile, cfg.eval.overlap)
            values = metric_bundle(restored, sharp, cfg.eval.crop_border)
            row = {
                "image": batch["path"][0],
                "sharp": batch["sharp_path"][0],
                **values,
            }
            rows.append(row)
            for key, value in values.items():
                totals[key] += value
            if args.save_images:
                destination = output_path_for_input(
                    Path(batch["path"][0]), Path(cfg.data.root), output / "images"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                tensor_to_image(restored.add(1).mul(0.5)).save(destination)
            max_images = int(getattr(cfg.eval, "max_images", 0))
            if max_images > 0 and len(rows) >= max_images:
                break
    summary = {key: value / max(len(rows), 1) for key, value in totals.items()}
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "count": len(rows),
        "blur_variant": str(cfg.data.blur_variant),
        "dataset_pair_sha256": data_audit["pair_sha256"]["test"],
        "color_space": getattr(cfg.eval, "color_space", "rgb"),
        "crop_border": int(cfg.eval.crop_border),
        "metric_protocol": getattr(cfg.eval, "metric_protocol", "float_rgb"),
        "image_quantization": (
            "round_to_uint8_grid"
            if getattr(cfg.eval, "metric_protocol", "float_rgb") == "gopro_matlab"
            else "none_float"
        ),
        "lpips_input_quantization": (
            "round_to_uint8_grid"
            if getattr(cfg.eval, "metric_protocol", "float_rgb") == "gopro_matlab"
            else "none_float"
        ),
        "lpips_input_range": "[-1,1]" if metric_bundle.lpips_model is not None else "disabled",
        "ssim_protocol": {"window": 11, "sigma": 1.5, "padding": "valid", "covariance": "population"},
        "aggregation": "per_image_then_arithmetic_mean",
        "lpips_backend": metric_bundle.lpips_backend,
        "perceptual_metric": metric_bundle.perceptual_metric,
        "perceptual_weights": metric_bundle.perceptual_weights,
        "perceptual_backbone": metric_bundle.perceptual_backbone,
        "lpips_version": metric_bundle.lpips_version,
        "lpips_calibrated": metric_bundle.lpips_calibrated,
        "tiled_inference": int(cfg.eval.tile),
        "runtime_config_sha256": config_fingerprint(cfg),
        "architecture_sha256": architecture_fingerprint(cfg),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_config_sha256": checkpoint.get("config_sha256"),
        "checkpoint_training_sha256": checkpoint.get("training_sha256"),
        **summary,
    }
    write_json(result, output / "metrics.json")
    print(result)


if __name__ == "__main__":
    main()
