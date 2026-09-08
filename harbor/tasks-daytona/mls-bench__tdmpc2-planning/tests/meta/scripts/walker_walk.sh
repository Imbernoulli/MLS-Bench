#!/bin/bash
cd tdmpc2 && python train.py \
    task=walker-walk \
    model_size=1 \
    steps=300000 \
    eval_freq=50000 \
    enable_wandb=false \
    save_video=false \
    save_agent=false \
    compile=false \
    seed=${SEED:-42}
