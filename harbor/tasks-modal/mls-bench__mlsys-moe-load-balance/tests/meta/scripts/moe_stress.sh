#!/bin/bash
# Stress config: 256 experts, 128 GPUs, 16 nodes, 384 replicas (1.5x),
# zipf_alpha=1.0, skew_ratio=0.95 — hidden config with pathological skew
# and large node hierarchy.

cd /workspace

python eplb/custom_eplb.py \
    --config stress-skew \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-.} \
    --num-trials 10 \
    --num-timing 20
