#!/bin/bash
python custom_irl.py \
    --env-id HalfCheetah-v4 \
    --seed ${SEED:-42} \
    --total-timesteps 1000000
