#!/bin/bash
# Working directory is already /workspace (package root)

SEED=${SEED:-42}

# Dataset config based on ENV label
case "${ENV}" in
  m4_monthly)   SP=Monthly ;;
  m4_quarterly) SP=Quarterly ;;
  m4_yearly)    SP=Yearly ;;
  *)            SP=Monthly ;;
esac

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --model DLinear \
  --data m4 \
  --root_path /data/m4 \
  --seasonal_patterns "$SP" \
  --model_id "m4_${SP}" \
  --features M \
  --enc_in 1 --dec_in 1 --c_out 1 \
  --d_model 512 --d_ff 512 \
  --e_layers 2 --d_layers 1 \
  --factor 3 \
  --n_heads 8 --dropout 0.1 \
  --train_epochs 10 --batch_size 16 \
  --patience 3 --learning_rate 0.001 \
  --loss SMAPE \
  --des "Exp_s${SEED}" --itr 1 \
  --seed $SEED
