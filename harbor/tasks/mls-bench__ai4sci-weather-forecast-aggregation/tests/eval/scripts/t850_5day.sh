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

python -u ClimaX/custom_forecast.py
