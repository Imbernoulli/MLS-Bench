#!/bin/bash
python cleanrl/custom_offpolicy_continuous.py \
    --env-id Reacher-v4 \
    --seed ${SEED:-42} \
    --total-timesteps 1000000
