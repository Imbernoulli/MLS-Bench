#!/bin/bash
python cleanrl/custom_value_discrete.py \
    --env-id Acrobot-v1 \
    --seed ${SEED:-42} \
    --total-timesteps 500000
