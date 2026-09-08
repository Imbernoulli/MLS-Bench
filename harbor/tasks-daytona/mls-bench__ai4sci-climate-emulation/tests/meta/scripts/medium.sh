#!/bin/bash
# Medium training run (100 epochs)

cd /workspace

# Drop this label's best_model from a previous agent iteration so an
# eval that never improves on it cannot silently score the old model.
rm -f "${OUTPUT_DIR}/best_model_${ENV:-default}.pt"
NUM_EPOCHS=100 EVAL_INTERVAL=10 \
python ClimSim/custom_emulator.py
