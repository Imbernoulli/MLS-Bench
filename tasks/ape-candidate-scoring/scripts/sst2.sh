#!/bin/bash
# ape-candidate-scoring: zero-shot SST-2 (2-class sentiment) with a
# FROZEN Qwen2.5-0.5B-Instruct. Same estimator surface; different dataset. The FIXED
# candidate pool is ranked by the agent's score_candidate(); the top pick is scored
# on the held-out TEST set by the calibrated forced-choice executor.
set -euo pipefail
cd /workspace/prompt-optimization-lab

python harness_scoring.py \
    --solution solution/scoring.py \
    --dataset sst2 \
    --seed ${SEED:-42}
