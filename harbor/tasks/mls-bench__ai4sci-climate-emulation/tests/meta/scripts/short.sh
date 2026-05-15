#!/bin/bash
# Short training run (30 epochs)

cd /workspace

NUM_EPOCHS=30 EVAL_INTERVAL=5 \
python ClimSim/custom_emulator.py
