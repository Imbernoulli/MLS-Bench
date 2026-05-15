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

python -u ClimaX/custom_forecast.py
