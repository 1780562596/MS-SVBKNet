from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run the paper's three independent seeds and aggregate test metrics")
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[3407, 3408, 3409])
    parser.add_argument("--name", default="full")
    parser.add_argument("--output-root", default="runs")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds contains duplicates")
    records = []
    for seed in args.seeds:
        overrides = [
            "--set", f"experiment.seed={seed}",
            "--set", f"experiment.name='{args.name}'",
            "--set", f"experiment.output_dir='{args.output_root}'",
        ]
        subprocess.run([
            sys.executable, "train.py", "--config", args.config,
            *overrides,
        ], check=True)
        checkpoint = Path(args.output_root) / args.name / f"seed_{seed}" / "checkpoints" / "best.pt"
        result_dir = Path(args.output_root) / args.name / f"seed_{seed}" / "test"
        subprocess.run([
            sys.executable, "evaluate.py", "--config", args.config, "--checkpoint", str(checkpoint),
            "--output", str(result_dir), *overrides,
        ], check=True)
        record = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
        record["seed"] = seed
        records.append(record)
    summary = {}
    for key in ("psnr", "ssim", "lpips"):
        values = [record[key] for record in records if key in record]
        if values:
            summary[key] = {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0}
    destination = Path(args.output_root) / args.name / "three_seed_summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "config": str(Path(args.config).resolve()),
                "seeds": args.seeds,
                "standard_deviation": "sample (n-1)",
                "runs": records,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary)


if __name__ == "__main__":
    main()
