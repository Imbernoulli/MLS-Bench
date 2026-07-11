#!/bin/bash
# summ-beam-repetition: decode the THREE FIXED domain settings (xsum / cnndm / samsum) with the
# FROZEN domain-matched summarizers, using the agent's config
# (solution/beam.py -> build_beam_config), then score corpus ROUGE-L F1 per setting (higher is
# better; the task score gmean's the 3 settings).
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/abstractive-summarization

python harness_beam.py \
    --solution solution/beam.py \
    --seed ${SEED:-42}
