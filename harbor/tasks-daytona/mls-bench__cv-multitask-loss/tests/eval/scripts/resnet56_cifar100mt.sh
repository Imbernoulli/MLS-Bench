#!/bin/bash
# Train ResNet-56 on CIFAR-100 with fine+coarse heads (~20 min on single GPU)
cd /workspace
python pytorch-vision/custom_mtl.py \
    --arch resnet56 \
    --data-root /data/cifar \
    --epochs 200 --batch-size 128 \
    --lr 0.1 --momentum 0.9 --weight-decay 5e-4 \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-./output}
