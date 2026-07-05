#!/bin/bash
# simp-beam-width: simplify each of the THREE FIXED test settings (asset / turk /
# wiki) with a FROZEN T5-base simplifier under a FIXED repetition/length decode
# config, varying ONLY the agent's beam WIDTH (solution/beamwidth.py ->
# build_num_beams), then score corpus SARI per setting (higher is better).
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/text-simplification

python harness_beamwidth.py \
    --solution solution/beamwidth.py \
    --seed ${SEED:-42}
