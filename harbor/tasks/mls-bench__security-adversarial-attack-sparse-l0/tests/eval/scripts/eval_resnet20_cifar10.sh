#!/bin/bash
# Working directory is /workspace (torchattacks package root).

python -u bench/run_eval.py \
  --arch resnet20 \
  --dataset cifar10 \
  --data-dir /data/cifar10 \
  --pixels 10 \
  --n-samples 100 \
  --batch-size 10 \
  --seed "${SEED:-42}"
