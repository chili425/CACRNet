import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from data import image_paths, jpeg_pillow
from metrics import calculate_psnr, calculate_psnrb, calculate_ssim
from model_factory import build_model


def pad_like_promptcir(tensor, multiple=32):
    height, width = tensor.shape[-2:]
    pad_h = (height // multiple + 1) * multiple - height
    pad_w = (width // multiple + 1) * multiple - width
    tensor = torch.cat([tensor, torch.flip(tensor, [2])], dim=2)
    tensor = tensor[:, :, :height + pad_h, :]
    tensor = torch.cat([tensor, torch.flip(tensor, [3])], dim=3)
    return tensor[:, :, :, :width + pad_w]


def tiled_restore(model, tensor, tile, overlap):
    _, _, height, width = tensor.shape
    tile = min(tile, height, width)
    tile -= tile % 16
    if tile <= 0:
        raise ValueError("Tile size must be at least 16")
    stride = max(16, tile - overlap)
    rows = list(range(0, max(height - tile, 0) + 1, stride))
    cols = list(range(0, max(width - tile, 0) + 1, stride))
    if not rows or rows[-1] != height - tile:
        rows.append(height - tile)
    if not cols or cols[-1] != width - tile:
        cols.append(width - tile)
    output = torch.zeros_like(tensor)
    weight = torch.zeros_like(tensor)
    for top in rows:
        for left in cols:
            patch = tensor[:, :, top:top + tile, left:left + tile]
            predictions, _ = model(patch)
            output[:, :, top:top + tile, left:left + tile] += predictions[0]
            weight[:, :, top:top + tile, left:left + tile] += 1
    return output / weight.clamp_min(1)


def to_tensor(image, device):
    array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float().div_(255).to(device)


def to_image(tensor):
    array = tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    return np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/eval_blind.yaml"))
    parser.add_argument("--gt-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    gt_dir = args.gt_dir or Path(cfg["gt_dir"])
    checkpoint = args.checkpoint or Path(cfg["checkpoint"])
    output_dir = args.output_dir or Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    if cfg.get("save_images", False):
        images_dir.mkdir(exist_ok=True)
    files = image_paths(gt_dir)
    torch.cuda.set_device(cfg["gpu"])
    device = torch.device(f"cuda:{cfg['gpu']}")
    model = build_model(cfg["width"])
    state = torch.load(checkpoint, map_location="cpu")
    if "net" in state and isinstance(state["net"], dict):
        state = state["net"]
    state = {name.removeprefix("module."): value for name, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    rng = random.Random(cfg["seed"])
    scores = []
    with (output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image", "qf", "psnr", "ssim", "psnrb", "input_psnr", "input_ssim", "input_psnrb"))
        with torch.no_grad():
            for index, path in enumerate(files, 1):
                with Image.open(path) as source:
                    gt_image = source.convert("RGB")
                qf = rng.randint(cfg["qf_min"], cfg["qf_max"])
                lq_image = jpeg_pillow(gt_image, qf, cfg["subsampling"])
                height, width = np.asarray(gt_image).shape[:2]
                padded = pad_like_promptcir(to_tensor(lq_image, device))
                restored = tiled_restore(model, padded, cfg["tile"], cfg["tile_overlap"])
                prediction = to_image(restored[:, :, :height, :width])
                gt = np.asarray(gt_image, dtype=np.uint8)
                lq = np.asarray(lq_image, dtype=np.uint8)
                values = (calculate_psnr(prediction, gt), calculate_ssim(prediction, gt),
                          calculate_psnrb(gt, prediction), calculate_psnr(lq, gt),
                          calculate_ssim(lq, gt), calculate_psnrb(gt, lq))
                scores.append(values)
                writer.writerow((path.name, qf, *(f"{value:.8f}" for value in values)))
                if cfg.get("save_images", False):
                    Image.fromarray(prediction).save(images_dir / f"{path.stem}.png")
                print(f"{index}/{len(files)} {path.name} QF={qf}: "
                      f"{values[0]:.4f}/{values[1]:.6f}/{values[2]:.4f}", flush=True)
    mean = np.asarray(scores).mean(axis=0)
    summary = (f"N={len(files)} seed={cfg['seed']} QF=[{cfg['qf_min']},{cfg['qf_max']}] "
               f"Pillow_{cfg['subsampling']}\n"
               f"PSNR={mean[0]:.4f} SSIM={mean[1]:.6f} PSNR-B={mean[2]:.4f}\n"
               f"Input PSNR={mean[3]:.4f} SSIM={mean[4]:.6f} PSNR-B={mean[5]:.4f}\n")
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
