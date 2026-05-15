#!/bin/bash
# Working directory is /workspace (torchattacks package root).

python -u bench/run_eval.py \
  --arch mobilenetv2_x1_0 \
  --dataset cifar100 \
  --data-dir /data/cifar100 \
  --pixels 10 \
  --n-samples 100 \
  --batch-size 10 \
  --seed "${SEED:-42}"
