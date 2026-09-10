#!/bin/bash
# Pretrain MobileNetV2 on FashionMNIST, then unlearn class 0
cd /workspace
python pytorch-vision/bench/unlearning/run_unlearning.py \
    --arch mobilenetv2 --dataset fmnist \
    --data-root /data/fmnist \
    --forget-class 0 \
    --pretrain-epochs 80 --unlearn-epochs 20 \
    --batch-size 128 \
    --lr 0.1 --momentum 0.9 --weight-decay 5e-4 \
    --seed ${SEED:-42}
