#!/bin/bash
# 2-layer MLP on CIFAR-10 (non-convex finite-sum problem)

cd /workspace

python opt-vr-bench/custom_vr.py \
    --problem mlp \
    --seed ${SEED:-42} \
    --output-dir "${OUTPUT_DIR:-./output}"
