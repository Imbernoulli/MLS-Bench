#!/bin/bash
python cleanrl/custom_offpolicy_continuous.py \
    --env-id Hopper-v4 \
    --seed ${SEED:-42} \
    --total-timesteps 1000000
