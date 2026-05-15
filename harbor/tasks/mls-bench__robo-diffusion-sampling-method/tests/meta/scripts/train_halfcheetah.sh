#!/bin/bash
set -e
cd /workspace/CleanDiffuser
SEED=${SEED:-42}
python pipelines/custom_sampling_method.py task=halfcheetah-medium-v2 mode=train seed=$SEED gradient_steps=100000 batch_size=256 log_interval=1000 save_interval=50000
python pipelines/custom_sampling_method.py task=halfcheetah-medium-v2 mode=inference seed=$SEED ckpt=100000 num_episodes=3 num_envs=50 num_candidates=50 use_ema=True
NFE=$(grep "^sampling_steps:" /workspace/CleanDiffuser/configs/custom/mujoco/mujoco.yaml | awk '{print $2}')
echo "NFE_METRICS sampling_steps=${NFE}"
