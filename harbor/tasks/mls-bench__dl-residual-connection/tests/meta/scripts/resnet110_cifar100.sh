#!/bin/bash
# Train ResNet-110 on CIFAR-100 (very deep, tests gradient flow)
cd /workspace
python pytorch-vision/custom_residual.py \
    --arch resnet110 --dataset cifar100 \
    --data-root /data/cifar \
    --epochs 200 --batch-size 128 \
    --lr 0.1 --momentum 0.9 --weight-decay 5e-4 \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-./output}
