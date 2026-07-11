#!/bin/bash
# summ-nucleus-topp: decode the THREE FIXED domain settings (xsum / cnndm / samsum) with the
# FROZEN domain-matched summarizers, using the agent's config
# (solution/topp.py -> build_top_p), then score corpus ROUGE-L F1 per setting (higher is
# better; the task score gmean's the 3 settings).
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/abstractive-summarization

python harness_topp.py \
    --solution solution/topp.py \
    --seed ${SEED:-42}
