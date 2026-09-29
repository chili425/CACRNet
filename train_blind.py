import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from data import BlindJPEGDataset
from model_factory import build_model


def psnr_loss(prediction, target):
    mse = F.mse_loss(prediction, target, reduction="none").mean((1, 2, 3))
    return -10.0 * torch.log10(1.0 / mse).mean()


def fft_loss(prediction, target):
    pred_fft = torch.fft.fft2(prediction, norm="ortho")
    gt_fft = torch.fft.fft2(target, norm="ortho")
    return F.l1_loss(pred_fft.real, gt_fft.real) + F.l1_loss(pred_fft.imag, gt_fft.imag)


def sobel_loss(prediction, target):
    def gray(tensor):
        return 0.299 * tensor[:, :1] + 0.587 * tensor[:, 1:2] + 0.114 * tensor[:, 2:3]

    kx = prediction.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8.0
    ky = prediction.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3) / 8.0
    pred_y, gt_y = gray(prediction), gray(target)
    return (F.l1_loss(F.conv2d(pred_y, kx, padding=1), F.conv2d(gt_y, kx, padding=1))
            + F.l1_loss(F.conv2d(pred_y, ky, padding=1), F.conv2d(gt_y, ky, padding=1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train_blind.yaml"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.set_device(cfg["gpu"])
    device = torch.device(f"cuda:{cfg['gpu']}")
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    dataset = BlindJPEGDataset(
        cfg["train_gt"], cfg["crop_size"], cfg["qf_min"], cfg["qf_max"],
        cfg["qf_step"], cfg["stage1_epochs"], cfg["subsampling"]
    )
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True,
                        drop_last=True, num_workers=cfg["num_workers"], pin_memory=True)
    if not len(loader):
        raise ValueError("Training set is smaller than one batch")
    model = build_model(cfg["width"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["min_learning_rate"]
    )
    start_epoch = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state["net"] if "net" in state else state, strict=True)
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_epoch = int(state["epoch"]) + 1

    total_steps = 0
    for epoch in range(start_epoch, cfg["epochs"]):
        dataset.set_epoch(epoch + 1)
        model.train()
        losses = []
        for lq, gt in loader:
            lq = lq.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            outputs, _ = model(lq, side_loss=True)
            gt_scales = [gt]
            for _ in range(len(outputs) - 1):
                gt_scales.append(F.interpolate(gt_scales[-1], scale_factor=0.5,
                                               mode="bilinear", align_corners=False))
            loss = psnr_loss(outputs[0], gt_scales[0])
            for scale in range(1, len(outputs)):
                loss = loss + psnr_loss(outputs[scale], gt_scales[scale]) / (2 ** scale)
            loss = (loss + cfg["fft_weight"] * fft_loss(outputs[0], gt)
                    + cfg["gradient_weight"] * sobel_loss(outputs[0], gt))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())
            total_steps += 1
            if args.max_steps and total_steps >= args.max_steps:
                print(f"Smoke test completed: loss={loss.item():.6f}", flush=True)
                return

        scheduler.step()
        print(f"epoch={epoch + 1} loss={sum(losses) / len(losses):.6f} "
              f"lr={optimizer.param_groups[0]['lr']:.8f}", flush=True)
        if (epoch + 1) % cfg["save_every"] == 0:
            torch.save(model.state_dict(), output / f"epoch_{epoch + 1}.pth")
            torch.save({"net": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "epoch": epoch},
                       output / f"epoch_{epoch + 1}.state")


if __name__ == "__main__":
    main()
