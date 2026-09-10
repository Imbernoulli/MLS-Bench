#!/bin/bash
python cleanrl/custom_intrinsic_exploration.py \
    --env-id Frostbite-v5 \
    --seed ${SEED:-42} \
    --total-timesteps 10000000
