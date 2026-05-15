#!/bin/bash
# Train VGG-16-BN on CIFAR-100 (~25 min on single GPU)
cd /workspace
python pytorch-vision/custom_reg.py \
    --arch vgg16bn --dataset cifar100 \
    --data-root /data/cifar \
    --epochs 200 --batch-size 128 \
    --lr 0.1 --momentum 0.9 --weight-decay 5e-4 \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-./output}
