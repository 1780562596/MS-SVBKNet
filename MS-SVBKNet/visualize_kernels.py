from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from ms_svbknet.config import load_config
from ms_svbknet.losses import credible_reblur_loss
from ms_svbknet.models import MSSVBKNet
from ms_svbknet.utils import load_checkpoint


def load_rgb(path: str, device):
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device).mul(2).sub(1)


def main():
    parser = argparse.ArgumentParser(description="Figure-12-style kernel and reliability visualization")
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="results/kernel_visualization.png")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    device = torch.device(cfg.experiment.device if torch.cuda.is_available() else "cpu")
    model = MSSVBKNet(cfg).to(device).eval()
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    blurred = load_rgb(args.image, device)
    with torch.no_grad():
        outputs = model(blurred)
        reblurred = model.reblur(outputs["restored"], outputs)
        _, reliability, error = credible_reblur_loss(reblurred, blurred, cfg.loss.reliability_alpha, True)
    basis = outputs["basis_kernels"]
    weights = outputs["kernel_weights"]
    if basis is None:
        raise RuntimeError("This visualization requires model.dictionary_mode=learned or fixed")
    n = basis.shape[0]
    columns = min(8, max(4, n))
    component_rows = math.ceil(n / columns)
    rows = component_rows * 2 + 2
    fig, axes = plt.subplots(rows, columns, figsize=(2.25 * columns, 2.15 * rows))
    for axis in axes.flat:
        axis.axis("off")
    for i in range(n):
        row, column = divmod(i, columns)
        axes[row, column].imshow(basis[i, 0].cpu(), cmap="hot")
        axes[row, column].set_title(f"basis k{i + 1}")
        axes[component_rows + row, column].imshow(weights[0, i].cpu(), cmap="viridis")
        axes[component_rows + row, column].set_title(f"weight w{i + 1}")

    visual_row = component_rows * 2
    images = [
        ((blurred[0].add(1).mul(0.5).permute(1, 2, 0).cpu()).clamp(0, 1), "blurred", None),
        ((outputs["restored"][0].add(1).mul(0.5).permute(1, 2, 0).cpu()).clamp(0, 1), "restored", None),
        ((reblurred[0].add(1).mul(0.5).permute(1, 2, 0).cpu()).clamp(0, 1), "reblurred", None),
        (error[0, 0].cpu(), "difference", "magma"),
        (reliability[0, 0].cpu(), "reliability", "viridis"),
    ]

    kernel = basis.shape[-1]
    coordinates = torch.arange(kernel, device=basis.device, dtype=basis.dtype) - (kernel - 1) / 2
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    radial_moment = (basis[:, 0] * (xx.square() + yy.square())).sum((1, 2))
    spread = (weights[0] * radial_moment[:, None, None]).sum(0)
    images.append((spread.cpu(), "kernel spread", "turbo"))
    for column, (data, title, cmap) in enumerate(images[:columns]):
        axes[visual_row, column].imshow(data, cmap=cmap)
        axes[visual_row, column].set_title(title)

    local_row = visual_row + 1
    flat = spread.flatten()
    ranks = [0, flat.numel() // 2, flat.numel() - 1]
    order = flat.argsort()
    for column, rank in enumerate(ranks):
        index = int(order[rank])
        y, x = divmod(index, spread.shape[1])
        local_kernel = (basis[:, 0] * weights[0, :, y, x, None, None]).sum(0)
        axes[local_row, column].imshow(local_kernel.cpu(), cmap="hot")
        axes[local_row, column].set_title(f"local K at ({x},{y})")
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
