#!/bin/bash
cd tdmpc2 && python train.py \
    task=cartpole-swingup \
    model_size=1 \
    steps=250000 \
    eval_freq=50000 \
    enable_wandb=false \
    save_video=false \
    save_agent=false \
    compile=false \
    seed=${SEED:-42}
