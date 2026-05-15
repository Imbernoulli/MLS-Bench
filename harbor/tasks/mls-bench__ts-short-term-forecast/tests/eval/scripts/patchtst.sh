#!/bin/bash
# Working directory is already /workspace (package root)
# Hyperparameters aligned with official TSLib per-pattern configs

SEED=${SEED:-42}

# Per-pattern d_model/d_ff tuned for M4 (small univariate datasets)
case "${ENV}" in
  m4_monthly)   SP=Monthly;   DM=32; DFF=32 ;;
  m4_quarterly) SP=Quarterly; DM=64; DFF=64 ;;
  m4_yearly)    SP=Yearly;    DM=16; DFF=32 ;;
  *)            SP=Monthly;   DM=32; DFF=32 ;;
esac

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --model PatchTST \
  --data m4 \
  --root_path /data/m4 \
  --seasonal_patterns "$SP" \
  --model_id "m4_${SP}" \
  --features M \
  --enc_in 1 --dec_in 1 --c_out 1 \
  --d_model $DM --d_ff $DFF \
  --e_layers 2 --d_layers 1 \
  --factor 3 \
  --n_heads 4 --dropout 0.1 \
  --train_epochs 10 --batch_size 16 \
  --patience 3 --learning_rate 0.001 \
  --loss SMAPE \
  --des "Exp_s${SEED}" --itr 1 \
  --seed $SEED
