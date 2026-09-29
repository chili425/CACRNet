# CACRNet: Compression-Aware Context Routing for JPEG Artifact Removal

This repository provides the PyTorch implementation of CACRNet for blind JPEG
artifact removal, including training and evaluation scripts and a pretrained
checkpoint.

## Requirements

The code was tested with Python 3.10, PyTorch 1.13.1, CUDA 11.7,
`mmcv-full` 1.7.0, and `mmcls` 0.25.0. See `requirements.txt` for the other
dependencies. 

## Data

Training uses DF2K (DIV2K and Flickr2K). Evaluation supports LIVE1, BSDS500,
ICB, and DIV2K. 

## Evaluation

python eval_blind.py --config configs/eval_blind.yaml

## Training

python train_blind.py --config configs/train_blind.yaml

