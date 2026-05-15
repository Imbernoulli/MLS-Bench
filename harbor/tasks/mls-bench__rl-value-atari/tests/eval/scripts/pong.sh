#!/bin/bash
python cleanrl/custom_value_atari.py \
    --env-id PongNoFrameskip-v4 \
    --seed ${SEED:-42} \
    --total-timesteps 5000000
