#!/bin/bash
# Long training run (200 epochs)

cd /workspace

NUM_EPOCHS=200 EVAL_INTERVAL=20 \
python ClimSim/custom_emulator.py
