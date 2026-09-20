from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .utils import seed_worker


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
BLUR_DIRS = {"blur", "blurry", "input", "blur_gamma", "blurred"}
SHARP_DIRS = ("sharp", "gt", "target", "clear", "ground_truth")


@dataclass(frozen=True)
class Pair:
    blur: Path
    sharp: Path


def _images(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def _replace_part(path: Path, index: int, value: str) -> Path:
    parts = list(path.parts)
    parts[index] = value
    return Path(*parts)


def discover_pairs(
    root: str | Path,
    split: str = "",
    index_file: str = "",
    blur_variant: str = "blur",
    enforce_index_variant: bool = False,
) -> List[Pair]:
    """Discover pairs without relying on independently sorted file lists.

    A CSV index may contain blur,sharp columns. Otherwise every blur path is
    resolved by replacing its semantic directory component, which prevents
    silent cross-scene pairing when filenames repeat across GoPro sequences.
    """
    root = Path(root).expanduser().resolve()
    base = root / split if split else root
    if index_file:
        index_path = Path(index_file)
        if not index_path.is_absolute():
            index_path = root / index_path
        if not index_path.is_file():
            raise FileNotFoundError(f"Pair index does not exist: {index_path}")
        pairs = []
        with index_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not {"blur", "sharp"}.issubset(reader.fieldnames):
                raise ValueError(f"Pair index must contain blur,sharp columns: {index_path}")
            for line_number, row in enumerate(reader, start=2):
                blur_text, sharp_text = (row.get("blur") or "").strip(), (row.get("sharp") or "").strip()
                if not blur_text or not sharp_text:
                    raise ValueError(f"Empty blur/sharp value in {index_path}:{line_number}")
                blur = Path(blur_text)
                sharp = Path(sharp_text)
                pairs.append(Pair(
                    (blur if blur.is_absolute() else root / blur).resolve(),
                    (sharp if sharp.is_absolute() else root / sharp).resolve(),
                ))
        if not pairs:
            raise ValueError(f"Pair index is empty: {index_path}")
        checked = validate_pairs(pairs)
        if enforce_index_variant:
            mismatched = [
                pair.blur
                for pair in checked
                if blur_variant.lower() not in {part.lower() for part in pair.blur.parts}
            ]
            if mismatched:
                examples = ", ".join(str(path) for path in mismatched[:5])
                raise ValueError(
                    f"CSV does not match data.blur_variant={blur_variant!r}; "
                    f"{len(mismatched)} paths differ, examples: {examples}"
                )
        return checked

    if not base.is_dir():
        raise FileNotFoundError(f"Dataset split does not exist: {base}")
    variant = blur_variant.strip().lower()
    if not variant:
        raise ValueError("blur_variant must be explicit for automatic discovery")
    pairs = []
    seen = set()
    missing = []
    for blur in _images(base):
        lower_parts = [part.lower() for part in blur.parts]
        semantic_indices = [i for i, part in enumerate(lower_parts) if part == variant]
        if not semantic_indices:
            continue
        idx = semantic_indices[-1]
        sharp = None
        for target_dir in SHARP_DIRS:
            candidate = _replace_part(blur, idx, target_dir)
            if candidate.exists():
                sharp = candidate
                break
        if sharp is None:
            missing.append(blur)
            continue
        blur, sharp = blur.resolve(), sharp.resolve()
        key = (str(blur), str(sharp))
        if key not in seen:
            pairs.append(Pair(blur, sharp))
            seen.add(key)
    if missing:
        examples = "\n".join(f"  - {path}" for path in missing[:10])
        suffix = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"{len(missing)} '{variant}' images have no matching sharp image under {base}:\n"
            f"{examples}{suffix}"
        )
    if not pairs:
        raise FileNotFoundError(
            f"No '{variant}' pairs found under {base}. Expected scene/{variant}/name.png + "
            "scene/sharp/name.png, or provide data.index_file CSV."
        )
    return validate_pairs(sorted(pairs, key=lambda p: str(p.blur)))


def validate_pairs(pairs: Sequence[Pair], inspect_sizes: bool = True) -> List[Pair]:
    checked = []
    seen_blur: Dict[Path, Path] = {}
    seen_exact = set()
    for pair in pairs:
        key = (pair.blur.resolve(), pair.sharp.resolve())
        if key in seen_exact:
            raise ValueError(f"Duplicate pair: {pair.blur} | {pair.sharp}")
        seen_exact.add(key)
        previous = seen_blur.get(key[0])
        if previous is not None and previous != key[1]:
            raise ValueError(f"Blur image maps to multiple targets: {pair.blur}")
        seen_blur[key[0]] = key[1]
        if not pair.blur.is_file() or not pair.sharp.is_file():
            raise FileNotFoundError(f"Missing pair: {pair.blur} | {pair.sharp}")
        if inspect_sizes:
            with Image.open(pair.blur) as a, Image.open(pair.sharp) as b:
                if a.size != b.size:
                    raise ValueError(f"Pair size mismatch: {pair.blur} {a.size} vs {pair.sharp} {b.size}")
        checked.append(pair)
    return checked


def scene_identifier(path: Path, blur_variant: str = "") -> str:
    """Return the directory above a semantic blur/sharp folder."""
    markers = set(BLUR_DIRS).union(SHARP_DIRS)
    if blur_variant:
        markers.add(blur_variant.lower())
    lower = [part.lower() for part in path.resolve().parts]
    indices = [index for index, part in enumerate(lower) if part in markers]
    if not indices:
        return str(path.resolve().parent)
    return str(Path(*path.resolve().parts[: indices[-1]]))


def pairs_fingerprint(pairs: Sequence[Pair], relative_to: Path | None = None) -> str:
    def stable_path(path: Path) -> str:
        resolved = path.resolve()
        if relative_to is not None:
            try:
                return resolved.relative_to(relative_to).as_posix()
            except ValueError:
                pass
        return resolved.as_posix()

    payload = [
        {"blur": stable_path(pair.blur), "sharp": stable_path(pair.sharp)}
        for pair in sorted(pairs, key=lambda item: (str(item.blur), str(item.sharp)))
    ]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def audit_pair_splits(
    split_pairs: Mapping[str, Sequence[Pair]],
    blur_variant: str = "blur",
    check_scene_leakage: bool = True,
) -> Dict[str, object]:
    """Fail on file or scene reuse across any supplied dataset partitions."""
    names = list(split_pairs)
    all_paths = [
        str(path.resolve())
        for pairs in split_pairs.values()
        for pair in pairs
        for path in (pair.blur, pair.sharp)
    ]
    try:
        common_root = Path(os.path.commonpath(all_paths)) if all_paths else None
    except ValueError:
        common_root = None
    report: Dict[str, object] = {
        "counts": {name: len(split_pairs[name]) for name in names},
        "pair_sha256": {
            name: pairs_fingerprint(split_pairs[name], relative_to=common_root) for name in names
        },
        "blur_variant": blur_variant,
        "scene_leakage_check": check_scene_leakage,
    }
    resolved = {
        name: {
            path.resolve()
            for pair in pairs
            for path in (pair.blur, pair.sharp)
        }
        for name, pairs in split_pairs.items()
    }
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = resolved[left].intersection(resolved[right])
            if overlap:
                examples = ", ".join(str(path) for path in sorted(overlap)[:5])
                raise ValueError(
                    f"Dataset leakage: {left}/{right} reuse {len(overlap)} files; examples: {examples}"
                )
    if check_scene_leakage:
        scene_sets = {
            name: {scene_identifier(pair.blur, blur_variant) for pair in pairs}
            for name, pairs in split_pairs.items()
        }
        report["scene_counts"] = {name: len(scenes) for name, scenes in scene_sets.items()}
        for left_index, left in enumerate(names):
            for right in names[left_index + 1:]:
                overlap = scene_sets[left].intersection(scene_sets[right])
                if overlap:
                    examples = ", ".join(sorted(overlap)[:5])
                    raise ValueError(
                        f"Scene leakage: {left}/{right} share {len(overlap)} scenes; examples: {examples}"
                    )
    return report


def validate_expected_scene_counts(report, data_config):
    """Check per-partition scene counts when a configuration specifies them."""
    for split in report["counts"]:
        expected = int(getattr(data_config, f"expected_{split}_scenes", 0))
        if expected:
            actual = report.get("scene_counts", {}).get(split)
            if actual != expected:
                raise ValueError(f"{split} scene count mismatch: expected {expected}, found {actual}")


def _read_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _pad_pair(blur: torch.Tensor, sharp: torch.Tensor, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    h, w = blur.shape[-2:]
    pad_h, pad_w = max(size - h, 0), max(size - w, 0)
    if pad_h or pad_w:
        mode = "reflect" if h > pad_h and w > pad_w else "replicate"
        padding = (0, pad_w, 0, pad_h)
        blur = F.pad(blur, padding, mode=mode)
        sharp = F.pad(sharp, padding, mode=mode)
    return blur, sharp


def paired_random_transform(
    blur: torch.Tensor,
    sharp: torch.Tensor,
    crop_size: int,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    blur, sharp = _pad_pair(blur, sharp, crop_size)
    h, w = blur.shape[-2:]
    top = int(torch.randint(0, h - crop_size + 1, (1,), generator=generator).item())
    left = int(torch.randint(0, w - crop_size + 1, (1,), generator=generator).item())
    blur = blur[:, top : top + crop_size, left : left + crop_size]
    sharp = sharp[:, top : top + crop_size, left : left + crop_size]
    if torch.rand((), generator=generator) < 0.5:
        blur, sharp = blur.flip(-1), sharp.flip(-1)
    if torch.rand((), generator=generator) < 0.5:
        blur, sharp = blur.flip(-2), sharp.flip(-2)
    k = int(torch.randint(0, 4, (1,), generator=generator).item())
    if k:
        blur, sharp = torch.rot90(blur, k, (-2, -1)), torch.rot90(sharp, k, (-2, -1))
    return blur.contiguous(), sharp.contiguous()


class PairedDeblurDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Pair],
        crop_size: int = 256,
        training: bool = True,
        seed: int = 0,
    ):
        self.pairs = list(pairs)
        self.crop_size = crop_size
        self.training = training
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        pair = self.pairs[index]
        blur, sharp = _read_rgb(pair.blur), _read_rgb(pair.sharp)
        if blur.shape != sharp.shape:
            raise RuntimeError(f"Decoded tensor mismatch: {pair.blur} | {pair.sharp}")
        if self.training:
            # Per-sample/epoch randomness is independent of worker scheduling.
            local_generator = torch.Generator()
            local_seed = (self.seed * 1_000_003 + self.epoch * 97_409 + index) % (2**63 - 1)
            local_generator.manual_seed(local_seed)
            blur, sharp = paired_random_transform(
                blur, sharp, self.crop_size, generator=local_generator
            )
        # The paper generator ends in tanh; use exactly one [0,1] -> [-1,1] conversion.
        return {
            "blur": blur.mul(2).sub(1),
            "sharp": sharp.mul(2).sub(1),
            "path": str(pair.blur),
            "sharp_path": str(pair.sharp),
        }


def make_loader(cfg, split: str, training: bool, seed: int) -> DataLoader:
    split_root = getattr(cfg.data, f"{split}_split", split)
    index_file = getattr(cfg.data, f"{split}_index", "")
    pairs = discover_pairs(
        cfg.data.root,
        split_root,
        index_file,
        blur_variant=str(cfg.data.blur_variant),
        enforce_index_variant=bool(getattr(cfg.data, "enforce_index_variant", False)),
    )
    dataset = PairedDeblurDataset(
        pairs,
        crop_size=cfg.data.crop_size,
        training=training,
        seed=seed,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size if training else 1,
        shuffle=training,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=training,
        persistent_workers=bool(getattr(cfg.data, "persistent_workers", False)) and cfg.data.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    # PyTorch does not expose this state through DataLoader.state_dict(). Keep a
    # public project-owned handle so checkpoints can reconstruct the next epoch.
    loader.repro_generator = generator
    return loader
