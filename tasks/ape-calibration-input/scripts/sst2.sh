#!/bin/bash
# ape-calibration-input: zero-shot sst2 with a FROZEN Qwen2.5-0.5B-Instruct. Only the
# calibration_inputs() surface is editable; the harness/executor/splits are fixed.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/prompt-optimization-lab

python harness_calibration.py \
    --solution solution/calibration.py \
    --dataset sst2 \
    --seed ${SEED:-42}
