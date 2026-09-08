#!/bin/bash
# Train DP-SGD on MNIST with epsilon=3.0.
#
# Hyperparameters (5 epochs, batch 256, lr 0.1, clip R=1.0) are tuned for the
# template's ReLU CNN, matching the ReLU baseline in Papernot et al. 2020
# "Tempered Sigmoids for Deep Learning with Differential Privacy"
# (arXiv:2007.14191) Table 4 (~96.6% @ ε≈3). Bu et al. 2023 Appendix G.1
# uses a tanh CNN to reach 98%+, which the fixed model architecture doesn't
# provide. Previous attempt with Bu's tanh-tuned setup (20 epochs, batch 512,
# lr 0.5, R=0.1) regressed ReLU accuracy from 96.3% to 94.9%.
cd /workspace

python opacus/custom_dpsgd.py \
    --dataset mnist \
    --epochs 5 \
    --batch-size 256 \
    --lr 0.1 \
    --max-grad-norm 1.0 \
    --target-epsilon 3.0 \
    --target-delta 1e-5 \
    --seed ${SEED:-42}
