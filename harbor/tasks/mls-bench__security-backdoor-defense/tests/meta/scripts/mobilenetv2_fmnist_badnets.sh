#!/bin/bash
# Train MobileNetV2 on FashionMNIST with BadNets backdoor, then run defense
cd /workspace
python pytorch-vision/bench/backdoor/run_backdoor_defense.py \
    --arch mobilenetv2 --dataset fmnist --data-root /data/fmnist \
    --trigger badnets --poison-fraction 0.08 \
    --epochs 100 --batch-size 128 --lr 0.1 --momentum 0.9 --weight-decay 5e-4 \
    --seed "${SEED:-42}"
