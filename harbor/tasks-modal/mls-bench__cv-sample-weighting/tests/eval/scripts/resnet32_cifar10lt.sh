#!/bin/bash
# Train ResNet-32 on Long-Tail CIFAR-10 (imbalance ratio=100)
cd /workspace
python pytorch-vision/custom_weighting.py \
    --arch resnet32 --dataset cifar10 \
    --imbalance-ratio 100 \
    --data-root /data/cifar \
    --epochs 200 --batch-size 128 \
    --lr 0.1 --momentum 0.9 --weight-decay 5e-4 \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-./output}
