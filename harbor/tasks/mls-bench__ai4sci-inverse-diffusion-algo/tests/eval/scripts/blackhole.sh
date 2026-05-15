#!/bin/bash
cd /workspace/InverseBench

python3 main.py \
    problem=blackhole \
    algorithm=custom \
    pretrain=blackhole \
    problem.data.root=/data/blackhole/test \
    problem.model.root=/data/blackhole/measure \
    seed=${SEED:-0} \
    wandb=False \
    tf32=True \
    num_samples=1 \
    exp_name=custom_blackhole
