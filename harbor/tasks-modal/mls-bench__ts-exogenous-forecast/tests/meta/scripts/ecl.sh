#!/bin/bash
# Working directory is already /workspace (package root)

SEED=${SEED:-42}

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model Custom \
  --data custom \
  --root_path /data/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96_MS \
  --features MS \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 321 --dec_in 321 --c_out 321 \
  --d_model 512 --d_ff 512 \
  --e_layers 2 --d_layers 1 \
  --n_heads 8 --dropout 0.1 \
  --train_epochs 10 --batch_size 32 \
  --patience 3 --learning_rate 0.0001 \
  --des "Exp_s${SEED}" --itr 1 \
  --seed $SEED
