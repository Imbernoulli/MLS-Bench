#!/bin/bash
python cleanrl/custom_onpolicy_continuous.py \
    --env-id Walker2d-v4 \
    --seed ${SEED:-42} \
    --total-timesteps 1000000
