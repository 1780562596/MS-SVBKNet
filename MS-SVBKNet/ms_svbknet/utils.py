from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def seed_everything(seed: int, deterministic: bool = True) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_manifest(cfg=None) -> Dict[str, Any]:
    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    manifest: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpus": gpu_names,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "dependencies": {},
    }
    for distribution in (
        "torchvision",
        "numpy",
        "Pillow",
        "PyYAML",
        "scikit-image",
        "lpips",
    ):
        try:
            manifest["dependencies"][distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            manifest["dependencies"][distribution] = None
    if cfg is not None:
        root = Path(cfg.data.root).expanduser()
        hashes = {}
        for split in ("train", "val", "test"):
            name = str(getattr(cfg.data, f"{split}_index", ""))
            if not name:
                continue
            path = Path(name)
            if not path.is_absolute():
                path = root / path
            if path.is_file():
                hashes[split] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
        manifest["data_indexes"] = hashes
    return manifest


def tensor_to_image(x: torch.Tensor) -> Image.Image:
    x = x.detach().float().clamp(0, 1)
    if x.ndim == 4:
        x = x[0]
    array = (x.permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def save_triplet(blur: torch.Tensor, restored: torch.Tensor, sharp: torch.Tensor, path: str | Path) -> None:
    images = [tensor_to_image(t) for t in (blur, restored, sharp)]
    canvas = Image.new("RGB", (sum(im.width for im in images), max(im.height for im in images)))
    x0 = 0
    for image in images:
        canvas.paste(image, (x0, 0))
        x0 += image.width
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def atomic_torch_save(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str | Path, map_location=None) -> Any:
    """Load this project's trusted checkpoint across PyTorch 2.1--2.6+."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def write_json(data: Dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def output_path_for_input(source: Path, input_root: Path, output_root: Path) -> Path:
    """Preserve relative directories so repeated frame names never overwrite."""
    source = source.resolve()
    input_root = input_root.resolve()
    try:
        relative = source.relative_to(input_root)
    except ValueError:
        relative = Path(source.name)
    return (output_root / relative).with_suffix(".png")


def append_csv(row: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def pad_to_multiple(x: torch.Tensor, multiple: int = 4):
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    mode = "reflect" if h > pad_h and w > pad_w else "replicate"
    return F.pad(x, (0, pad_w, 0, pad_h), mode=mode), (h, w)


@torch.no_grad()
def tiled_restore(model, image: torch.Tensor, tile: int = 0, overlap: int = 64) -> torch.Tensor:
    """Restore an image, optionally using overlap-add tiling without RGB seams."""
    padded, original_hw = pad_to_multiple(image, 4)
    if tile <= 0 or max(padded.shape[-2:]) <= tile:
        result = model(padded)["restored"]
        return result[..., : original_hw[0], : original_hw[1]]

    if overlap >= tile:
        raise ValueError("overlap must be smaller than tile")
    if padded.shape[-2] < tile or padded.shape[-1] < tile:
        extra_h = max(tile - padded.shape[-2], 0)
        extra_w = max(tile - padded.shape[-1], 0)
        mode = "reflect" if padded.shape[-2] > extra_h and padded.shape[-1] > extra_w else "replicate"
        padded = F.pad(padded, (0, extra_w, 0, extra_h), mode=mode)
    b, c, h, w = padded.shape
    if b != 1:
        raise ValueError("Tiled inference currently requires batch size 1")
    stride = tile - overlap
    ys = list(range(0, max(h - tile, 0) + 1, stride))
    xs = list(range(0, max(w - tile, 0) + 1, stride))
    if not ys or ys[-1] != h - tile:
        ys.append(max(h - tile, 0))
    if not xs or xs[-1] != w - tile:
        xs.append(max(w - tile, 0))

    output = torch.zeros((b, c, h, w), device=image.device, dtype=torch.float32)
    weight_sum = torch.zeros_like(output)
    wy = torch.hann_window(tile, periodic=False, device=image.device).clamp_min(1e-3)
    window = (wy[:, None] * wy[None, :])[None, None]
    for y in ys:
        for x in xs:
            patch = padded[..., y : y + tile, x : x + tile]
            pred = model(patch)["restored"].float()
            output[..., y : y + tile, x : x + tile] += pred * window
            weight_sum[..., y : y + tile, x : x + tile] += window
    output = output / weight_sum.clamp_min(1e-6)
    return output[..., : original_hw[0], : original_hw[1]]
