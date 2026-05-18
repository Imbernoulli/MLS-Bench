#!/bin/bash
# Train VGG-16-BN on CIFAR-100 with membership defense loss.
# Paper-aligned recipe (RelaxLoss, Chen et al., ICLR 2022):
# 300 epochs, SGD+momentum, step-LR at [150, 225], no augmentation,
# wd=1e-4, batch=128. Training set is the 25K half-split (member_subset)
# — matches the 25K CIFAR-100 target partition used by RelaxLoss-style
# membership-inference evaluations. This regime yields measurable MIA leakage
# for the ERM control in this benchmark.
# Official config:
# https://github.com/DingfanChen/RelaxLoss/blob/main/source/cifar/defense/configs/default.yml
cd /workspace
python pytorch-vision/run_membership_defense.py \
    --arch vgg16bn --dataset cifar100 \
    --data-root /data/cifar \
    --epochs 300 --batch-size 128 \
    --lr 0.1 --momentum 0.9 --weight-decay 1e-4 \
    --schedule-milestones 150 225 --schedule-gamma 0.1 \
    --seed "${SEED:-42}"
