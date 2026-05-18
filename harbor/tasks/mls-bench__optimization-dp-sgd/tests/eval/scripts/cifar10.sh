#!/bin/bash
# Train DP-SGD on CIFAR-10 with epsilon=3.0
cd /workspace

python opacus/custom_dpsgd.py \
    --dataset cifar10 \
    --epochs 30 \
    --batch-size 256 \
    --lr 0.05 \
    --max-grad-norm 1.0 \
    --target-epsilon 3.0 \
    --target-delta 1e-5 \
    --seed ${SEED:-42}
