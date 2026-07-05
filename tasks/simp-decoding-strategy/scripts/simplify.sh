#!/bin/bash
# simp-decoding-strategy: simplify each of the THREE FIXED test settings (asset /
# turk / wiki) with a FROZEN T5-base simplifier using the agent's DECODING STRATEGY
# (solution/strategy.py -> build_strategy: "sample" / "topp" / "beam"), then score
# corpus SARI per setting (higher is better).
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/text-simplification

python harness_strategy.py \
    --solution solution/strategy.py \
    --seed ${SEED:-42}
