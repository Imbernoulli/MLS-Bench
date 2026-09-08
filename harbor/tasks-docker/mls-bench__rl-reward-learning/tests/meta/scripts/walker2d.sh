#!/bin/bash
python custom_irl.py \
    --env-id Walker2d-v4 \
    --seed ${SEED:-42} \
    --total-timesteps 1000000
