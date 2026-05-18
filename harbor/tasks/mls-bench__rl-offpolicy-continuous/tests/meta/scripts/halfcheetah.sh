#!/bin/bash
python cleanrl/custom_offpolicy_continuous.py \
    --env-id HalfCheetah-v4 \
    --seed ${SEED:-42} \
    --total-timesteps 1000000
