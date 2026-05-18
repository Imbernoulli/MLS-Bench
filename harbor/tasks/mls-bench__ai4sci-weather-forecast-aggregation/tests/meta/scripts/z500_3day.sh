#!/bin/bash
# ERA5 geopotential height at 500hPa, 3-day (72h) lead time

cd /workspace

export OUT_VAR="geopotential_500"
export PREDICT_RANGE=72
export MAX_EPOCHS=100
export BATCH_SIZE=64
export LR=5e-4
export WARMUP_STEPS=5000
export PATIENCE=20

python -u ClimaX/custom_forecast.py
