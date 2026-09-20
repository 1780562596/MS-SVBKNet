from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ms_svbknet.config import architecture_fingerprint, config_fingerprint, load_config, verify_checkpoint_architecture
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


def read_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).mul(2).sub(1)


def main():
    parser = argparse.ArgumentParser(description="Deblur one image or a directory")
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results/restored")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg.experiment.seed, bool(cfg.experiment.deterministic))
    device = torch.device(cfg.experiment.device if torch.cuda.is_available() else "cpu")
    model = MSSVBKNet(cfg).to(device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    verify_checkpoint_architecture(checkpoint, cfg)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()
    source, output = Path(args.input), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    input_root = source.parent if source.is_file() else source
    paths = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
    with torch.no_grad():
        for path in tqdm(paths):
            restored = tiled_restore(model, read_image(path).to(device), cfg.eval.tile, cfg.eval.overlap)
            destination = output_path_for_input(path, input_root, output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            tensor_to_image(restored.add(1).mul(0.5)).save(destination)
    write_json(
        {
            "count": len(paths),
            "input": str(source.resolve()),
            "output": str(output.resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_config_sha256": checkpoint.get("config_sha256"),
            "runtime_config_sha256": config_fingerprint(cfg),
            "architecture_sha256": architecture_fingerprint(cfg),
            "tile": int(cfg.eval.tile),
            "overlap": int(cfg.eval.overlap),
        },
        output / "inference_manifest.json",
    )


if __name__ == "__main__":
    main()
