#!/bin/bash
# Working directory is already /workspace (package root)

SEED=${SEED:-42}
# No official TSLib PatchTST imputation script; aligned with DLinear imputation style

case "${ENV}" in
  ETTh1)
    DATA=ETTh1; ROOT=/data/ETT-small/; DPATH=ETTh1.csv; EI=7; DI=7; CO=7; MID=ETTh1_mask_0.25 ;;
  Weather)
    DATA=custom; ROOT=/data/weather/; DPATH=weather.csv; EI=21; DI=21; CO=21; MID=Weather_mask_0.25 ;;
  ECL)
    DATA=custom; ROOT=/data/electricity/; DPATH=electricity.csv; EI=321; DI=321; CO=321; MID=ECL_mask_0.25 ;;
esac

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --model PatchTST \
  --data "$DATA" \
  --root_path "$ROOT" \
  --data_path "$DPATH" \
  --model_id "$MID" \
  --features M \
  --seq_len 96 --label_len 0 --pred_len 96 \
  --mask_rate 0.25 \
  --enc_in $EI --dec_in $DI --c_out $CO \
  --d_model 128 --d_ff 128 \
  --e_layers 2 --d_layers 1 \
  --factor 3 --top_k 5 \
  --n_heads 8 --dropout 0.1 \
  --train_epochs 10 --batch_size 16 \
  --patience 3 --learning_rate 0.001 \
  --des "Exp_s${SEED}" --itr 1 \
  --seed $SEED
