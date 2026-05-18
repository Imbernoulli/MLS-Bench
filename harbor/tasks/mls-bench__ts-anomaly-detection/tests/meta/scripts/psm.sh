#!/bin/bash
# Working directory is already /workspace (package root)

SEED=${SEED:-42}

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --model Custom \
  --data PSM \
  --root_path /data/PSM \
  --model_id PSM \
  --seq_len 100 --pred_len 100 \
  --enc_in 25 --c_out 25 \
  --d_model 512 --d_ff 512 \
  --e_layers 2 --d_layers 1 \
  --n_heads 8 --dropout 0.1 \
  --anomaly_ratio 1 \
  --train_epochs 3 --batch_size 32 \
  --patience 3 --learning_rate 0.0001 \
  --des "Exp_s${SEED}" --itr 1 \
  --seed $SEED
