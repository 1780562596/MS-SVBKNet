from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def sharp_pattern(size: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    image = Image.new("RGB", (size, size), color=(24, 30, 44))
    draw = ImageDraw.Draw(image)
    for _ in range(18):
        x0, y0 = rng.integers(0, size - 8, size=2)
        x1 = int(min(size - 1, x0 + rng.integers(4, max(5, size // 3))))
        y1 = int(min(size - 1, y0 + rng.integers(4, max(5, size // 3))))
        color = tuple(int(value) for value in rng.integers(48, 256, size=3))
        if rng.random() < 0.5:
            draw.rectangle((int(x0), int(y0), x1, y1), fill=color)
        else:
            draw.ellipse((int(x0), int(y0), x1, y1), fill=color)
    draw.line((0, size - 1, size - 1, 0), fill=(255, 255, 255), width=2)
    return image


def shifted(array: np.ndarray, dx: int, dy: int) -> np.ndarray:
    height, width = array.shape[:2]
    padded = np.pad(array, ((abs(dy), abs(dy)), (abs(dx), abs(dx)), (0, 0)), mode="edge")
    y0 = abs(dy) + dy
    x0 = abs(dx) + dx
    return padded[y0 : y0 + height, x0 : x0 + width]


def motion_blur(image: Image.Image, radius: int, vertical: bool) -> Image.Image:
    array = np.asarray(image, dtype=np.float32)
    frames = []
    for offset in range(-radius, radius + 1):
        frames.append(shifted(array, 0 if vertical else offset, offset if vertical else 0))
    blurred = np.mean(frames, axis=0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(blurred, mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny paired dataset for end-to-end smoke tests")
    parser.add_argument("--output", default="data/toy")
    parser.add_argument("--size", type=int, default=72)
    parser.add_argument("--train", type=int, default=4)
    parser.add_argument("--val", type=int, default=2)
    parser.add_argument("--test", type=int, default=2)
    args = parser.parse_args()
    root = Path(args.output)
    for split, count, seed_offset in (("train", args.train, 0), ("val", args.val, 100), ("test", args.test, 200)):
        for index in range(count):
            scene = root / split / f"scene_{index:03d}"
            blur_dir, sharp_dir = scene / "blur", scene / "sharp"
            blur_dir.mkdir(parents=True, exist_ok=True)
            sharp_dir.mkdir(parents=True, exist_ok=True)
            sharp = sharp_pattern(args.size, seed_offset + index)
            blur = motion_blur(sharp, radius=1 + index % 3, vertical=bool(index % 2))
            sharp.save(sharp_dir / "000001.png")
            blur.save(blur_dir / "000001.png")
    print(f"Toy dataset written to {root.resolve()}")


if __name__ == "__main__":
    main()
