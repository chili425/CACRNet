# CACRNet: Compression-Aware Context Routing for JPEG Artifact Removal

This repository provides the PyTorch implementation of CACRNet for blind JPEG
artifact removal, including training and evaluation scripts and a pretrained
checkpoint.

## Requirements

The code was tested with Python 3.10, PyTorch 1.13.1, CUDA 11.7,
`mmcv-full` 1.7.0, and `mmcls` 0.25.0. See `requirements.txt` for the other
dependencies. The Fourier-RWKV CUDA extension is compiled on first use, so
`nvcc` is required.

## Data

Training uses DF2K (DIV2K and Flickr2K). Evaluation supports LIVE1, BSDS500,
ICB, and DIV2K. Point the configuration files to flat directories of clean
images; JPEG inputs are generated with 4:4:4 chroma sampling and quality
factors in `[10, 70]`.

## Evaluation

Set `gt_dir` in `configs/eval_blind.yaml`, then run:

```bash
python eval_blind.py --config configs/eval_blind.yaml
```

The included `weights/cacrnet_blind_epoch1030.pth` is the width-144 checkpoint
used for blind evaluation. Results are saved under `results/`.

## Training

Set `train_gt` in `configs/train_blind.yaml`, then run:

```bash
python train_blind.py --config configs/train_blind.yaml
```

The training recipe follows the paper settings but is not guaranteed to
reproduce the included historical checkpoint exactly.

## Acknowledgements

The model builds on Fourier-RWKV. The metric implementation includes code
from IJCN; its license is provided in `third_party/IJCN_LICENSE`.
