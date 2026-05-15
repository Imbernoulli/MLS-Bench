#!/bin/bash
# Working directory is already /workspace (package root)

SEED=${SEED:-42}

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --model Custom \
  --data UEA \
  --root_path /data/Handwriting/ \
  --model_id Handwriting \
  --d_model 128 --d_ff 256 \
  --e_layers 3 --d_layers 1 \
  --n_heads 16 --dropout 0.1 \
  --train_epochs 100 --batch_size 16 \
  --patience 10 --learning_rate 0.001 \
  --des "Exp_s${SEED}" --itr 1 \
  --seed $SEED
