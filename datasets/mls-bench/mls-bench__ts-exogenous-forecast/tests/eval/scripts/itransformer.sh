#!/bin/bash
# Working directory is already /workspace (package root)

SEED=${SEED:-42}

case "${ENV}" in
  ETTh1)
    DATA=ETTh1; ROOT=/data/ETT-small/; DPATH=ETTh1.csv; EI=7; DI=7; CO=7; MID=ETTh1_96_96_MS ;;
  Weather)
    DATA=custom; ROOT=/data/weather/; DPATH=weather.csv; EI=21; DI=21; CO=21; MID=Weather_96_96_MS ;;
  ECL)
    DATA=custom; ROOT=/data/electricity/; DPATH=electricity.csv; EI=321; DI=321; CO=321; MID=ECL_96_96_MS ;;
esac

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model iTransformer \
  --data "$DATA" \
  --root_path "$ROOT" \
  --data_path "$DPATH" \
  --model_id "$MID" \
  --features MS \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in $EI --dec_in $DI --c_out $CO \
  --d_model 512 --d_ff 512 \
  --e_layers 2 --d_layers 1 \
  --n_heads 8 --dropout 0.1 \
  --train_epochs 10 --batch_size 32 \
  --patience 3 --learning_rate 0.0001 \
  --des "Exp_s${SEED}" --itr 1 \
  --seed $SEED
