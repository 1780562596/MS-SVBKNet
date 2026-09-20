from __future__ import annotations

import argparse
import json
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def matched_inputs(directories):
    """Require the same relative image paths in all supplied branches."""
    inventories = {}
    for label, directory in directories.items():
        root = Path(directory)
        if not root.is_dir():
            raise ValueError(f"Missing image directory: {root}")
        inventories[label] = {
            path.relative_to(root).as_posix(): str(path)
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        if not inventories[label]:
            raise ValueError(f"No images found: {root}")
    reference = set(next(iter(inventories.values())))
    if any(set(items) != reference for items in inventories.values()):
        raise ValueError("Input branches must contain matching relative image paths")
    return {label: [items[key] for key in sorted(reference)] for label, items in inventories.items()}


def run_comparison(model, inputs, datasets, classes, imgsz=640, conf=0.25, val_conf=0.001, iou=0.70):
    """Use one frozen detector and explicit parameters for every branch."""
    summary = {}
    for label, images in inputs.items():
        results = model.predict(
            source=images, classes=classes, imgsz=imgsz, conf=conf, iou=iou,
            save=True, verbose=False,
        )
        confidences = [float(score) for result in results for score in result.boxes.conf.cpu()]
        counts = [int(result.boxes.cls.numel()) for result in results]
        summary[label] = {
            "images": len(results),
            "mean_confidence": sum(confidences) / max(len(confidences), 1),
            "vehicles_per_image": sum(counts) / max(len(counts), 1),
        }
        if datasets.get(label):
            metrics = model.val(
                data=datasets[label], classes=classes, imgsz=imgsz,
                conf=val_conf, iou=iou, verbose=False,
            )
            summary[label]["labeled_evaluation"] = {
                "map50": float(metrics.box.map50),
                "map50_95": float(metrics.box.map),
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
            }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Frozen YOLO comparison of matched traffic images")
    parser.add_argument("--blur-dir", required=True)
    parser.add_argument("--restored-dir", required=True)
    parser.add_argument("--sharp-dir", default="", help="Sharp-reference branch for the paper's three-way comparison")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--output", default="results/yolo_summary.json")
    parser.add_argument("--classes", nargs="+", type=int, default=[2, 3, 5, 7])
    parser.add_argument("--blur-data", default="")
    parser.add_argument("--restored-data", default="")
    parser.add_argument("--sharp-data", default="")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--val-conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.70)
    args = parser.parse_args()
    if args.imgsz <= 0 or any(not 0 <= value <= 1 for value in (args.conf, args.val_conf, args.iou)):
        parser.error("imgsz must be positive and confidence/IoU thresholds must lie in [0, 1]")
    if args.sharp_data and not args.sharp_dir:
        parser.error("--sharp-data requires --sharp-dir")
    directories = {"blur": args.blur_dir, "restored": args.restored_dir}
    if args.sharp_dir:
        directories["sharp"] = args.sharp_dir
    inputs = matched_inputs(directories)
    datasets = {label: getattr(args, f"{label}_data") for label in directories}
    if any(datasets.values()) and not all(datasets.values()):
        parser.error("Labeled comparison requires a dataset YAML for every supplied branch")
    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install the optional package: pip install ultralytics") from exc
    summary = run_comparison(
        YOLO(args.weights), inputs, datasets, args.classes,
        args.imgsz, args.conf, args.val_conf, args.iou,
    )
    summary["protocol"] = {
        "weights": args.weights, "ultralytics_version": ultralytics.__version__,
        "classes": args.classes, "imgsz": args.imgsz, "visualization_conf": args.conf,
        "validation_conf": args.val_conf, "nms_iou": args.iou,
        "detector_frozen": True, "matched_images_per_branch": len(inputs["blur"]),
        "dataset_yamls": datasets,
        "annotation_note": "Dataset YAMLs must reference the same matched frames and shared ground-truth labels.",
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
