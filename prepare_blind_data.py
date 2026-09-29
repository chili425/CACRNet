import argparse
import csv
import random
from pathlib import Path

from PIL import Image

from data import image_paths, jpeg_pillow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--qf-min", type=int, default=10)
    parser.add_argument("--qf-max", type=int, default=70)
    parser.add_argument("--subsampling", choices=("default", "444"), default="444")
    args = parser.parse_args()
    files = image_paths(args.gt_dir)
    rng = random.Random(args.seed)
    lq_dir = args.output_dir / "LQ"
    lq_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image", "qf", "lq_file"))
        for path in files:
            qf = rng.randint(args.qf_min, args.qf_max)
            with Image.open(path) as source:
                lq = jpeg_pillow(source.convert("RGB"), qf, args.subsampling)
            destination = lq_dir / f"{path.stem}.png"
            lq.save(destination)
            writer.writerow((path.name, qf, str(destination.relative_to(args.output_dir))))
    print(f"Generated {len(files)} JPEG inputs in {lq_dir}")


if __name__ == "__main__":
    main()
