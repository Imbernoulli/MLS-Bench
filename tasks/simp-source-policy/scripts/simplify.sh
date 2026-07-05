#!/bin/bash
# simp-source-policy: rewrite each of the THREE FIXED simplification test settings
# (asset / turk / wiki) by the agent's SOURCE POLICY (solution/policy.py ->
# build_policy), then score corpus SARI per setting (higher is better). Proves SARI
# is monotone / un-gameable: copy-input < truncate < greedy < tuned-beam T5.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/text-simplification

python harness_policy.py \
    --solution solution/policy.py \
    --seed ${SEED:-42}
