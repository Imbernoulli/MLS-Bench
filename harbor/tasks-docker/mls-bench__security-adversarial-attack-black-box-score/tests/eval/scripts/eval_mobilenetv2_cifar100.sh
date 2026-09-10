#!/bin/bash

python -u bench/run_eval.py \
  --arch mobilenetv2_x1_0 \
  --dataset cifar100 \
  --data-dir /data/cifar100 \
  --eps 0.03137255 \
  --n-samples 200 \
  --n-queries 1000 \
  --batch-size 50 \
  --seed "${SEED:-42}"
