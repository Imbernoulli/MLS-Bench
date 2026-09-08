#!/bin/bash
# ERA5 temperature at 850hPa, 5-day (120h) lead time

cd /workspace

export OUT_VAR="temperature_850"
export PREDICT_RANGE=120
export MAX_EPOCHS=100
export BATCH_SIZE=64
export LR=5e-4
export WARMUP_STEPS=5000
export PATIENCE=20

# Scope checkpoints per label: all three labels run concurrently and share
# OUTPUT_DIR, so an unscoped best_model.pt gets clobbered across labels.
export OUTPUT_DIR="${OUTPUT_DIR:-output}/${ENV:-t850-5day}"
mkdir -p "$OUTPUT_DIR"
# Drop any stale checkpoint from a previous run/iteration.
rm -f "${OUTPUT_DIR}/best_model.pt"

python -u ClimaX/custom_forecast.py
