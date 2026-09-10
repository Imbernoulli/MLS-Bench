#!/bin/bash
# Adversarial training: VGG-11-BN on CIFAR-10 (eps=8/255)
# Working directory is /workspace (torchattacks package root).

python -u bench/run_adv_train.py \
  --arch vgg11_bn \
  --dataset cifar10 \
  --data-dir /data/cifar10 \
  --epochs 80 \
  --batch-size 128 \
  --lr 0.1 \
  --weight-decay 5e-4 \
  --eps 0.03137255 \
  --alpha 0.00784314 \
  --attack-steps 10 \
  --eval-attack-steps 50 \
  --eval-alpha 0.00784314 \
  --eval-restarts 3 \
  --seed "${SEED:-42}"
