#!/bin/bash
set -e
cd /workspace/CleanDiffuser
# Verifier-only: build pipelines/_verifier_custom_policy.py — the agent's file plus the
# race fix when the pre-fix block is still present. The agent's file
# itself is never modified, keeping the pristine guard valid on
# verifier re-runs (see _runtime_patch.sh).
source "$(dirname "${BASH_SOURCE[0]}")/_runtime_patch.sh"
SEED=${SEED:-42}
python pipelines/_verifier_custom_policy.py task=halfcheetah-medium-v2 mode=train seed=$SEED gradient_steps=1000000 batch_size=256 log_interval=1000 save_interval=100000
python pipelines/_verifier_custom_policy.py task=halfcheetah-medium-v2 mode=inference seed=$SEED ckpt=1000000 num_episodes=3 num_envs=50 num_candidates=50 use_ema=True
