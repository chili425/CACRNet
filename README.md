# CACRNet: blind JPEG restoration

This is a standalone reproduction package extracted from the experimental
UCMNet development tree. It contains only the blind single-JPEG model,
training/evaluation entry points, JPEG data generation, configuration, the
CUDA kernel source, and the checkpoint used for the reported blind evaluation.
No datasets, generated outputs, ablation scripts, or development backups are
included.

## Method and code

- `model/cacrnet.py`: CACRNet network. The public classes use the paper's
  names: `CACRNet`, `ContextualCodebookBank`, `SpatialChannelInteraction`,
  and `CompressionAwareGatedFusion` (CAGF). The checkpoint's internal module
  attribute names are deliberately unchanged, so strict loading remains
  possible.
- `model/backbone.py` and `model/modules_frwkv/`: Fourier-RWKV backbone and
  its CUDA extension source.
- `data.py`, `train_blind.py`, `eval_blind.py`, `prepare_blind_data.py`:
  blind JPEG data, training, evaluation, and optional offline input generation.
- `weights/cacrnet_blind_epoch1030.pth`: width-144 evaluation checkpoint.

## Environment

The tested server environment is Python 3.10.16, PyTorch 1.13.1+cu117,
torchvision 0.14.1+cu117, CUDA toolkit 11.7, MMCV-full 1.7.0, and MMCls
0.25.0. The other package versions are listed in `requirements.txt`.
Install a PyTorch/CUDA-matched `mmcv-full` wheel and ensure `nvcc` is
available. The Fourier-RWKV extension compiles on first import. If the CUDA
runtime libraries are not discovered automatically, run with:

```bash
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
```

No compiled `.so`, build cache, or conda environment is bundled.

## Data

Training uses the flat DF2K ground-truth image directory (DIV2K and
Flickr2K). Patches are cropped to 128x128, randomly flipped and rotated,
then JPEG-compressed online using Pillow 4:4:4 (`subsampling=0`). The first
1000 epochs sample QF from `{10,20,30,40,50,60,70}`; subsequent epochs
sample integer QF uniformly from `[10,70]`. No compressed training dataset
needs to be stored.

For an optional, deterministic set of compressed validation inputs:

```bash
python prepare_blind_data.py --gt-dir /path/to/GT --output-dir /path/to/paired
```

This writes `LQ/*.png` and `manifest.csv`; it does not duplicate the GT images.

## Blind evaluation

Replace the example dataset path in `configs/eval_blind.yaml` and select the
GPU, then run:

```bash
python eval_blind.py --config configs/eval_blind.yaml
```

The default configuration evaluates LIVE1 with seed 2024, one uniformly
sampled integer QF in `[10,70]` per image, Pillow 4:4:4 compression,
flip-padding to a multiple of 32, tiled inference (512/64), and full-image
RGB PSNR/SSIM/PSNR-B. Image order is sorted. The expected LIVE1 result with
the bundled checkpoint is approximately 33.03 dB PSNR, 0.917 SSIM, and
32.43 dB PSNR-B. Per-image values are saved to `results/LIVE1/per_image.csv`.
Pass `--gt-dir`, `--checkpoint`, or `--output-dir` to override the config.

For BSDS500, ICB, and DIV2K, point `gt_dir` to a flat image directory and
set a separate `output_dir`. Use the exact benchmark image list to compare
against paper results.

## Training

Replace the example training-data path in `configs/train_blind.yaml`, then run:

```bash
python train_blind.py --config configs/train_blind.yaml
```

The recipe uses Adam (0.9, 0.99), initial LR 2e-4, cosine decay to 2e-5
over 2000 epochs, and PSNR, FFT, and Sobel losses. Weights and full resume
states are saved every 10 epochs. `--resume path/to/epoch.state` continues
training; `--max-steps 1` performs a startup smoke test.

The supplied `epoch1030` checkpoint and network snapshot are the original
evaluation artifacts. The historical training command for this checkpoint
was not preserved in full, so the included clean training recipe follows the
manuscript protocol but is **not claimed to regenerate identical weights**.
Do not report a new training result as a replication of that checkpoint
without rerunning and documenting the complete experiment.

## Provenance

The network retains portions of the original UCMNet implementation and
Fourier-RWKV backbone. PSNR-B/SSIM metric functions were extracted from the
IJCN utility used in the original evaluation; its license is in
`third_party/IJCN_LICENSE`. The model class renames affect Python symbols,
not learned weights or forward computations.
