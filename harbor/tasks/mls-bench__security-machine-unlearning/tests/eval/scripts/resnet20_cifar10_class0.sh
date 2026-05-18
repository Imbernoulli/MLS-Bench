#!/bin/bash
# Pretrain ResNet-20 on CIFAR-10, then unlearn class 0
cd /workspace
python pytorch-vision/bench/unlearning/run_unlearning.py \
    --arch resnet20 --dataset cifar10 \
    --data-root /data/cifar \
    --forget-class 0 \
    --pretrain-epochs 80 --unlearn-epochs 20 \
    --batch-size 128 \
    --lr 0.1 --momentum 0.9 --weight-decay 5e-4 \
    --seed ${SEED:-42}
