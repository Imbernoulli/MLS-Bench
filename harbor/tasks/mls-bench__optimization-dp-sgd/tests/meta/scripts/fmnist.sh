#!/bin/bash
# Train DP-SGD on Fashion-MNIST with epsilon=3.0.
#
# Hyperparameters (5 epochs, batch 256, lr 0.1, clip R=1.0) are tuned for the
# template's ReLU CNN and epsilon=3.0 harness. Alternative tanh-network recipes
# from the literature use a different fixed architecture than this task exposes.
cd /workspace

python opacus/custom_dpsgd.py \
    --dataset fmnist \
    --epochs 5 \
    --batch-size 256 \
    --lr 0.1 \
    --max-grad-norm 1.0 \
    --target-epsilon 3.0 \
    --target-delta 1e-5 \
    --seed ${SEED:-42}
