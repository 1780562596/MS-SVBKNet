from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ms_svbknet.config import load_config
from ms_svbknet.data import audit_pair_splits, discover_pairs, validate_expected_scene_counts
from ms_svbknet.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit pair counts, fingerprints, and cross-split leakage"
    )
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--output", default="data_audit.json")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    split_pairs = {}
    for split in ("train", "val", "test"):
        split_root = getattr(cfg.data, f"{split}_split", split)
        index_file = getattr(cfg.data, f"{split}_index", "")
        split_pairs[split] = discover_pairs(
            cfg.data.root,
            split_root,
            index_file,
            blur_variant=str(cfg.data.blur_variant),
            enforce_index_variant=bool(getattr(cfg.data, "enforce_index_variant", False)),
        )
    report = audit_pair_splits(
        split_pairs,
        blur_variant=str(cfg.data.blur_variant),
        check_scene_leakage=bool(getattr(cfg.data, "enforce_scene_disjoint", True)),
    )
    validate_expected_scene_counts(report, cfg.data)
    expected_trainval = int(getattr(cfg.data, "expected_trainval_pairs", 0))
    expected_test = int(getattr(cfg.data, "expected_test_pairs", 0))
    expected_trainval_scenes = int(getattr(cfg.data, "expected_trainval_scenes", 0))
    expected_test_scenes = int(getattr(cfg.data, "expected_test_scenes", 0))
    actual_trainval = len(split_pairs["train"]) + len(split_pairs["val"])
    if expected_trainval and actual_trainval != expected_trainval:
        raise RuntimeError(
            f"train+val count mismatch: expected {expected_trainval}, found {actual_trainval}"
        )
    if expected_test and len(split_pairs["test"]) != expected_test:
        raise RuntimeError(
            f"test count mismatch: expected {expected_test}, found {len(split_pairs['test'])}"
        )
    scene_counts = report.get("scene_counts", {})
    actual_trainval_scenes = int(scene_counts.get("train", 0)) + int(scene_counts.get("val", 0))
    if expected_trainval_scenes and actual_trainval_scenes != expected_trainval_scenes:
        raise RuntimeError(
            f"train+val scene count mismatch: expected {expected_trainval_scenes}, "
            f"found {actual_trainval_scenes}"
        )
    if expected_test_scenes and int(scene_counts.get("test", 0)) != expected_test_scenes:
        raise RuntimeError(
            f"test scene count mismatch: expected {expected_test_scenes}, "
            f"found {scene_counts.get('test', 0)}"
        )
    report["expected_counts"] = {
        "train_plus_val": expected_trainval,
        "test": expected_test,
        "train_plus_val_scenes": expected_trainval_scenes,
        "test_scenes": expected_test_scenes,
    }
    write_json(report, Path(args.output))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
