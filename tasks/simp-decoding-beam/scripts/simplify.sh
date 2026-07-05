#!/bin/bash
# simp-decoding-beam: simplify each of the THREE FIXED test settings (asset / turk /
# wiki) with a FROZEN T5-base simplifier using the agent's BEAM / REPETITION decode
# config (solution/beam.py -> build_beam_config), then score corpus SARI per setting
# (higher is better).
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/text-simplification

python harness_beam.py \
    --solution solution/beam.py \
    --seed ${SEED:-42}
