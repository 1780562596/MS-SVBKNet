from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def select_validation_scenes(scenes, seed=3407, val_scenes=None, val_fraction=None):
    """Select whole scenes; default to the paper's four validation scenes."""
    scenes = sorted(set(scenes))
    if len(scenes) < 2:
        raise ValueError("At least two scenes are required")
    if val_scenes is not None and val_fraction is not None:
        raise ValueError("Choose a scene count or a fraction, not both")
    if val_fraction is not None:
        if not 0 < val_fraction < 1:
            raise ValueError("val_fraction must be strictly between 0 and 1")
        count = min(len(scenes) - 1, max(1, round(len(scenes) * val_fraction)))
    else:
        count = 4 if val_scenes is None else val_scenes
        if not 1 <= count < len(scenes):
            raise ValueError("Validation scene count must leave nonempty train and val sets")
    random.Random(seed).shuffle(scenes)
    return set(scenes[:count])


def write_index(path: Path, pairs, root: Path):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["blur", "sharp"])
        writer.writeheader()
        for pair in pairs:
            writer.writerow({"blur": pair.blur.relative_to(root), "sharp": pair.sharp.relative_to(root)})


def main():
    from ms_svbknet.data import audit_pair_splits, discover_pairs, scene_identifier

    parser = argparse.ArgumentParser(description="Create leakage-free GoPro train/val CSVs by scene")
    parser.add_argument("--root", required=True)
    parser.add_argument("--source-split", default="train")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--val-scenes", type=int, help="Validation scene count (default: 4)")
    selection.add_argument("--val-fraction", type=float, help="Alternative validation fraction")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--blur-variant",
        choices=("blur", "blur_gamma"),
        default="blur_gamma",
        help="Choose exactly one GoPro input variant; variants are never mixed",
    )
    parser.add_argument(
        "--expected-total",
        type=int,
        default=2103,
        help="Expected GoPro training-pair count; use 0 to disable",
    )
    parser.add_argument("--expected-scenes", type=int, default=22)
    parser.add_argument("--train-output", default="train_index.csv")
    parser.add_argument("--val-output", default="val_index.csv")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    pairs = discover_pairs(root, args.source_split, blur_variant=args.blur_variant)
    if args.expected_total and len(pairs) != args.expected_total:
        raise RuntimeError(
            f"Expected {args.expected_total} pairs, found {len(pairs)} for {args.blur_variant}"
        )
    scenes = sorted({scene_identifier(pair.blur, args.blur_variant) for pair in pairs})
    if args.expected_scenes and len(scenes) != args.expected_scenes:
        raise RuntimeError(
            f"Expected {args.expected_scenes} scenes, found {len(scenes)} for {args.blur_variant}"
        )
    val_scenes = select_validation_scenes(scenes, args.seed, args.val_scenes, args.val_fraction)
    train_pairs = [
        pair for pair in pairs if scene_identifier(pair.blur, args.blur_variant) not in val_scenes
    ]
    val_pairs = [
        pair for pair in pairs if scene_identifier(pair.blur, args.blur_variant) in val_scenes
    ]
    audit_pair_splits(
        {"train": train_pairs, "val": val_pairs}, blur_variant=args.blur_variant
    )
    train_output = Path(args.train_output)
    val_output = Path(args.val_output)
    if not train_output.is_absolute():
        train_output = root / train_output
    if not val_output.is_absolute():
        val_output = root / val_output
    train_output.parent.mkdir(parents=True, exist_ok=True)
    val_output.parent.mkdir(parents=True, exist_ok=True)
    write_index(train_output, train_pairs, root)
    write_index(val_output, val_pairs, root)
    print(f"train={len(train_pairs)} val={len(val_pairs)} scenes={len(scenes)}")


if __name__ == "__main__":
    main()
