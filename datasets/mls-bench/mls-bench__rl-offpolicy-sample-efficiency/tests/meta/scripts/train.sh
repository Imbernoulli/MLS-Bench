#!/bin/bash
# Agent's custom algorithm training script
cd fast_td3
python custom_algorithm.py \
    --env_name "${ENV}" \
    --seed "${SEED:-1}" \
    --total_timesteps 100000 \
    --num_envs 128 \
    --device_rank 0
