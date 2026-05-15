#!/bin/bash
# Working directory is already /workspace (package root)

SEED=${SEED:-42}
# Hyperparameters aligned with official TSLib per-pattern scripts

case "${ENV}" in
  m4_monthly)   SP=Monthly;   DM=32; DFF=32 ;;
  m4_quarterly) SP=Quarterly; DM=64; DFF=64 ;;
  m4_yearly)    SP=Yearly;    DM=16; DFF=32 ;;
  *)            SP=Monthly;   DM=32; DFF=32 ;;
esac

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --model TimesNet \
  --data m4 \
  --root_path /data/m4 \
  --seasonal_patterns "$SP" \
  --model_id "m4_${SP}" \
  --features M \
  --enc_in 1 --dec_in 1 --c_out 1 \
  --d_model $DM --d_ff $DFF \
  --e_layers 2 --d_layers 1 \
  --factor 3 --top_k 5 \
  --n_heads 8 --dropout 0.1 \
  --train_epochs 10 --batch_size 16 \
  --patience 3 --learning_rate 0.001 \
  --loss SMAPE \
  --des "Exp_s${SEED}" --itr 1 \
  --seed $SEED
