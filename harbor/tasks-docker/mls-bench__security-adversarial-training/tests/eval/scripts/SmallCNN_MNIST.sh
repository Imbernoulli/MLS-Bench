#!/bin/bash
# Adversarial training: SmallCNN on MNIST (eps=0.3)
# Working directory is /workspace (torchattacks package root).

python -u bench/run_adv_train.py \
  --arch smallcnn \
  --dataset mnist \
  --data-dir /data/mnist \
  --epochs 25 \
  --batch-size 128 \
  --lr 0.01 \
  --weight-decay 0.0 \
  --eps 0.3 \
  --alpha 0.01 \
  --attack-steps 40 \
  --eval-attack-steps 50 \
  --eval-alpha 0.01 \
  --eval-restarts 3 \
  --seed "${SEED:-42}"
