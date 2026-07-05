#!/bin/bash
# simp-nucleus-sampling: simplify each of the THREE FIXED test settings (asset /
# turk / wiki) with a FROZEN T5-base simplifier using SAMPLING (do_sample=True,
# num_beams=1, temperature=1.0 FIXED) restricted to the agent's NUCLEUS (top-p)
# (solution/nucleus.py -> build_top_p), then score corpus SARI per setting (higher
# is better).
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/text-simplification

python harness_nucleus.py \
    --solution solution/nucleus.py \
    --seed ${SEED:-42}
