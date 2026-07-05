#!/bin/bash
# simp-decoding-temperature: simplify each of the THREE FIXED test settings (asset /
# turk / wiki) with a FROZEN T5-base simplifier using SAMPLING (do_sample=True,
# num_beams=1 FIXED) at the agent's TEMPERATURE (solution/temperature.py ->
# build_temperature), then score corpus SARI per setting (higher is better).
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/text-simplification

python harness_temperature.py \
    --solution solution/temperature.py \
    --seed ${SEED:-42}
