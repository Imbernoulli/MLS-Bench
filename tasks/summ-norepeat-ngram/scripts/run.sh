#!/bin/bash
# summ-norepeat-ngram: decode the THREE FIXED domain settings (xsum / cnndm / samsum) with the
# FROZEN domain-matched summarizers, using the agent's config
# (solution/norepeat.py -> build_norepeat_size), then score corpus ROUGE-L F1 per setting (higher is
# better; the task score gmean's the 3 settings).
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/abstractive-summarization

python harness_norepeat.py \
    --solution solution/norepeat.py \
    --seed ${SEED:-42}
