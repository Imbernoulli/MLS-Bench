#!/bin/bash
# Train ResNet-20 on CIFAR-10 with membership defense loss.
# Paper-aligned recipe (RelaxLoss, Chen et al., ICLR 2022):
# 300 epochs, SGD+momentum, step-LR at [150, 225], no augmentation, wd=1e-4.
# Official config:
# https://github.com/DingfanChen/RelaxLoss/blob/main/source/cifar/defense/configs/default.yml
cd /workspace
python pytorch-vision/run_membership_defense.py \
    --arch resnet20 --dataset cifar10 \
    --data-root /data/cifar \
    --epochs 300 --batch-size 128 \
    --lr 0.1 --momentum 0.9 --weight-decay 1e-4 \
    --schedule-milestones 150 225 --schedule-gamma 0.1 \
    --seed "${SEED:-42}"
