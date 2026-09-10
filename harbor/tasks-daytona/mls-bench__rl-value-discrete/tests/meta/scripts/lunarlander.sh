#!/bin/bash
python cleanrl/custom_value_discrete.py \
    --env-id LunarLander-v2 \
    --seed ${SEED:-42} \
    --total-timesteps 500000 \
    --buffer-size 100000 \
    --learning-starts 10000 \
    --train-frequency 4 \
    --exploration-fraction 0.3 \
    --end-e 0.02
