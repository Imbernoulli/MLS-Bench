#!/bin/bash
# VGG-11-BN on CIFAR-100 (9.8M params) — large model, harder dataset
cd /workspace
python pytorch-vision/custom_compressor.py \
    --model vgg11 \
    --dataset cifar100 \
    --batch-size 128 \
    --epochs 200 \
    --lr 0.05 \
    --weight-decay 5e-4 \
    --warmup-epochs 5 \
    --compress-ratio 0.01 \
    --seed ${SEED:-42}
