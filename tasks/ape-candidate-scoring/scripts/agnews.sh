#!/bin/bash
# ape-candidate-scoring: zero-shot AG News (4-class) with a FROZEN Qwen2.5-0.5B-
# Instruct. The candidate instruction POOL and the argmax search rule are FIXED; the
# agent's score_candidate() is the ESTIMATOR that RANKS candidates on the dev set.
# The single top-ranked candidate is scored on the HELD-OUT TEST set by the
# calibrated forced-choice executor. Everything but the estimator is fixed.
set -euo pipefail
cd /workspace/prompt-optimization-lab

python harness_scoring.py \
    --solution solution/scoring.py \
    --dataset agnews \
    --seed ${SEED:-42}
