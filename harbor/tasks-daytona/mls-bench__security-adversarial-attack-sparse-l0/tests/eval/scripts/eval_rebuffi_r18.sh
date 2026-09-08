#!/bin/bash
# Working directory is /workspace (torchattacks package root).
# Sparse-RS canonical L0 setting (Croce et al., AAAI 2022): k=24,
# untargeted, adversarially-robust CIFAR-10 target (RobustBench L2).
# This is the exact model the paper's App. A.5 cites (l2-AT PreActResNet-18).

python -u bench/run_eval.py \
  --model-name Rebuffi2021Fixing_R18_cutmix_ddpm \
  --model-dir /data/robustbench_models \
  --data-dir /data/cifar10 \
  --pixels 24 \
  --n-samples 150 \
  --batch-size 50 \
  --seed "${SEED:-42}"
