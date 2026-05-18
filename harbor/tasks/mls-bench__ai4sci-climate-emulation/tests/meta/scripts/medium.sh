#!/bin/bash
# Medium training run (100 epochs)

cd /workspace

NUM_EPOCHS=100 EVAL_INTERVAL=10 \
python ClimSim/custom_emulator.py
