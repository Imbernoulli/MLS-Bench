#!/bin/bash
python train_safe_rl.py \
    --algo CustomLag \
    --env-id SafetyPointButton1-v0 \
    --seed ${SEED:-42} \
    --total-steps 2000000 \
    --device cuda:0
