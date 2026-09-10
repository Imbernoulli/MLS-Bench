#!/bin/bash
# ResNet-56 on CIFAR-10 (0.85M params) — deeper model, tests compression at scale
cd /workspace
python pytorch-vision/custom_compressor.py \
    --model resnet56 \
    --dataset cifar10 \
    --batch-size 128 \
    --epochs 200 \
    --lr 0.1 \
    --weight-decay 5e-4 \
    --warmup-epochs 5 \
    --compress-ratio 0.01 \
    --seed ${SEED:-42}
