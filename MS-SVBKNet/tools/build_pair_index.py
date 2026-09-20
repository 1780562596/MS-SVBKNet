from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ms_svbknet.data import IMAGE_SUFFIXES


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an auditable blur/sharp CSV from matching relative paths")
    parser.add_argument("--root", required=True, help="Dataset root used for relative CSV paths")
    parser.add_argument("--blur-dir", required=True)
    parser.add_argument("--sharp-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    blur_dir = Path(args.blur_dir).expanduser().resolve()
    sharp_dir = Path(args.sharp_dir).expanduser().resolve()
    rows = []
    missing = []
    for blur in sorted(path for path in blur_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES):
        relative = blur.relative_to(blur_dir)
        sharp = sharp_dir / relative
        if not sharp.is_file():
            missing.append(str(relative))
            continue
        try:
            rows.append({"blur": blur.relative_to(root), "sharp": sharp.relative_to(root)})
        except ValueError as exc:
            raise SystemExit("Both image directories must be located under --root") from exc
    if missing:
        preview = "\n".join(missing[:10])
        raise SystemExit(f"Missing {len(missing)} sharp counterparts. First entries:\n{preview}")
    if not rows:
        raise SystemExit("No image pairs found")
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["blur", "sharp"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} pairs to {output}")


if __name__ == "__main__":
    main()
