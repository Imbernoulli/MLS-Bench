#!/bin/bash
set -e
cd /workspace/CleanDiffuser
SEED=${SEED:-42}
python pipelines/custom_guidance.py task=halfcheetah-medium-v2 mode=train seed=$SEED ++diffusion_gradient_steps=100000 ++classifier_gradient_steps=100000 batch_size=256 log_interval=1000 save_interval=50000
python pipelines/custom_guidance.py task=halfcheetah-medium-v2 mode=inference seed=$SEED ++ckpt=100000 num_episodes=10 num_envs=10 use_ema=True
