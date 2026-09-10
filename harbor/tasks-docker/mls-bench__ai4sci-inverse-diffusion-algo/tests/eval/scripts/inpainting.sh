#!/bin/bash
mkdir -p exps/inference/ffhq256-inpaint/Custom/custom_inpaint

python3 main.py \
    problem=ffhq256_inpaint \
    algorithm=custom \
    pretrain=ffhq256 \
    problem.prior=/workspace/InverseBench/checkpoints/ffhq256.pt \
    problem.data.root=/data/ffhq256 \
    seed=${SEED:-42} \
    wandb=False \
    tf32=True \
    num_samples=1 \
    exp_name=custom_inpaint
