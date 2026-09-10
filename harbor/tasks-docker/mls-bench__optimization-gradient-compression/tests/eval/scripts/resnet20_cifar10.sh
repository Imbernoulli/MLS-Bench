#!/bin/bash
# ResNet-20 on CIFAR-10 (0.27M params) — small model, standard dataset
cd /workspace
python pytorch-vision/custom_compressor.py \
    --model resnet20 \
    --dataset cifar10 \
    --batch-size 128 \
    --epochs 200 \
    --lr 0.1 \
    --weight-decay 5e-4 \
    --warmup-epochs 5 \
    --compress-ratio 0.01 \
    --seed ${SEED:-42}
