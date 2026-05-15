#!/bin/bash
# Working directory is /workspace (torchattacks package root).

python -u bench/run_eval.py \
  --arch resnet20 \
  --dataset cifar100 \
  --data-dir /data/cifar100 \
  --eps 0.00784314 \
  --n-samples 1000 \
  --batch-size 100 \
  --seed "${SEED:-42}"
