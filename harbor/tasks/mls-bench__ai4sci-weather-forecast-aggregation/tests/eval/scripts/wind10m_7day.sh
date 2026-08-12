#!/bin/bash
# ERA5 10m u-component of wind, 7-day (168h) lead time

cd /workspace

export OUT_VAR="10m_u_component_of_wind"
export PREDICT_RANGE=168
export MAX_EPOCHS=100
export BATCH_SIZE=64
export LR=5e-4
export WARMUP_STEPS=5000
export PATIENCE=20

# Scope checkpoints per label: all three labels run concurrently and share
# OUTPUT_DIR, so an unscoped best_model.pt gets clobbered across labels.
export OUTPUT_DIR="${OUTPUT_DIR:-output}/${ENV:-wind10m-7day}"
mkdir -p "$OUTPUT_DIR"
# Drop any stale checkpoint from a previous run/iteration.
rm -f "${OUTPUT_DIR}/best_model.pt"

python -u ClimaX/custom_forecast.py
