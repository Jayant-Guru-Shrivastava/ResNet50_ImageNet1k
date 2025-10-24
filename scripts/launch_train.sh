#!/usr/bin/env bash
set -euo pipefail
DATA=${1:-/mnt/imagenet-1k}
OUT=${2:-runs/r50d-imnet}
python src/train_imagenet.py   --data "$DATA"   --model resnet50d   --epochs 200   --batch-size 256   --workers 8   --aa rand-m9-mstd0.5   --mixup 0.2 --cutmix 1.0 --label-smoothing 0.1 --reprob 0.25   --lr 0.1 --warmup-epochs 5   --amp --ema   --output "$OUT"