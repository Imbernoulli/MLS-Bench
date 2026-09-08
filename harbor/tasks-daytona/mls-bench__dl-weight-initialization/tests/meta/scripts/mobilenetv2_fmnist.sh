#!/bin/bash
# Train MobileNetV2 on FashionMNIST (~15 min on single GPU)
cd /workspace
python pytorch-vision/custom_init.py \
    --arch mobilenetv2 --dataset fmnist \
    --data-root /data/fmnist \
    --epochs 200 --batch-size 128 \
    --lr 0.1 --momentum 0.9 --weight-decay 5e-4 \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-./output}
