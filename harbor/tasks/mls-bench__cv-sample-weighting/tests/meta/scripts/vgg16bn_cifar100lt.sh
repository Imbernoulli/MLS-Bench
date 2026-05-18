#!/bin/bash
# Train VGG-16-BN on Long-Tail CIFAR-100 (imbalance ratio=50)
cd /workspace
python pytorch-vision/custom_weighting.py \
    --arch vgg16bn --dataset cifar100 \
    --imbalance-ratio 50 \
    --data-root /data/cifar \
    --epochs 200 --batch-size 128 \
    --lr 0.1 --momentum 0.9 --weight-decay 5e-4 \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-./output}
