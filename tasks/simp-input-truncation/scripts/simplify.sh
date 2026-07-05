#!/bin/bash
# simp-input-truncation: simplify each of the THREE FIXED test settings (asset /
# turk / wiki) with a FROZEN T5-base simplifier under a FIXED strong beam decode,
# varying ONLY the agent's ENCODER-SIDE input-truncation budget
# (solution/truncation.py -> build_max_input_tokens), then score corpus SARI per
# setting (higher is better).
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/text-simplification

python harness_truncation.py \
    --solution solution/truncation.py \
    --seed ${SEED:-42}
