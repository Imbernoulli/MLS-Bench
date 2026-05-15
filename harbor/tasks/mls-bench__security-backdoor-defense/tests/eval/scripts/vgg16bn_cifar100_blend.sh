#!/bin/bash
# Train VGG-16-BN on CIFAR-100 with Blend backdoor, then run defense
cd /workspace
python pytorch-vision/bench/backdoor/run_backdoor_defense.py \
    --arch vgg16bn --dataset cifar100 --data-root /data/cifar \
    --trigger blend --poison-fraction 0.01 \
    --epochs 100 --batch-size 128 --lr 0.1 --momentum 0.9 --weight-decay 5e-4 \
    --seed "${SEED:-42}"
