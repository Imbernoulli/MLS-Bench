#!/bin/bash
# Working directory is already /workspace (package root)

SEED=${SEED:-42}
# Hyperparameters aligned with official TSLib TimeMixer_M4 script

case "${ENV}" in
  m4_monthly)   SP=Monthly;   DFF=32 ;;
  m4_quarterly) SP=Quarterly; DFF=64 ;;
  m4_yearly)    SP=Yearly;    DFF=32 ;;
  *)            SP=Monthly;   DFF=32 ;;
esac

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --model TimeMixer \
  --data m4 \
  --root_path /data/m4 \
  --seasonal_patterns "$SP" \
  --model_id "m4_${SP}" \
  --features M \
  --enc_in 1 --dec_in 1 --c_out 1 \
  --d_model 32 --d_ff $DFF \
  --e_layers 4 --d_layers 1 \
  --factor 3 \
  --train_epochs 50 --batch_size 128 \
  --patience 20 --learning_rate 0.01 \
  --down_sampling_layers 1 \
  --down_sampling_method avg \
  --down_sampling_window 2 \
  --loss SMAPE \
  --des "Exp_s${SEED}" --itr 1 \
  --seed $SEED
