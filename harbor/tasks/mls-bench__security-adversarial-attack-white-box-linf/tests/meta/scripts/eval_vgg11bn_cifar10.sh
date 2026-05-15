#!/bin/bash
# Working directory is /workspace (torchattacks package root).

python -u bench/run_eval.py \
  --arch vgg11_bn \
  --dataset cifar10 \
  --data-dir /data/cifar10 \
  --eps 0.00784314 \
  --n-samples 1000 \
  --batch-size 100 \
  --seed "${SEED:-42}"
