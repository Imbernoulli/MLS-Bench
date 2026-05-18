#!/bin/bash
mkdir -p exps/inference/inv-scatter-linear/Custom/custom_inv_scatter

python3 main.py \
    problem=inv-scatter \
    algorithm=custom \
    pretrain=inv-scatter \
    problem.prior=/workspace/InverseBench/checkpoints/inv-scatter-5m.pt \
    problem.data.root=/data/inv-scatter-test \
    seed=${SEED:-42} \
    wandb=False \
    tf32=True \
    num_samples=1 \
    exp_name=custom_inv_scatter
