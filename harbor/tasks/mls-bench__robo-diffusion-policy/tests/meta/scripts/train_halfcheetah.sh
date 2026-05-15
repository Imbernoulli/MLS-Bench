#!/bin/bash
set -e
cd /workspace/CleanDiffuser
SEED=${SEED:-42}
python pipelines/custom_policy.py task=halfcheetah-medium-v2 mode=train seed=$SEED gradient_steps=1000000 batch_size=256 log_interval=1000 save_interval=100000
python pipelines/custom_policy.py task=halfcheetah-medium-v2 mode=inference seed=$SEED ckpt=1000000 num_episodes=3 num_envs=50 num_candidates=50 use_ema=True
