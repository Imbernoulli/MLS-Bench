#!/bin/bash
# Working directory is /workspace (torchattacks package root).
# Sparse-RS canonical L0 setting: k=24, untargeted, RobustBench L2-robust
# CIFAR-10 target (Engstrom et al. 2019).

python -u bench/run_eval.py \
  --model-name Engstrom2019Robustness \
  --model-dir /data/robustbench_models \
  --data-dir /data/cifar10 \
  --pixels 24 \
  --n-samples 150 \
  --batch-size 50 \
  --seed "${SEED:-42}"
