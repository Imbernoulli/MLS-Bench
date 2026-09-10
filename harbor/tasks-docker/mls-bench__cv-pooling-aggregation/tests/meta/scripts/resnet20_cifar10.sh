#!/bin/bash
# Train ResNet-20 on CIFAR-10 (~10 min on single GPU)
cd /workspace
python pytorch-vision/custom_pool.py \
    --arch resnet20 --dataset cifar10 \
    --data-root /data/cifar \
    --epochs 200 --batch-size 128 \
    --lr 0.1 --momentum 0.9 --weight-decay 5e-4 \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-./output}
