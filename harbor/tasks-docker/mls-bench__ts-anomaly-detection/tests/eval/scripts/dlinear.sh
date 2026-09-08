#!/bin/bash
# Working directory is already /workspace (package root)

SEED=${SEED:-42}
# Hyperparameters aligned with official TSLib scripts

case "${ENV}" in
  PSM)  DATA=PSM;  ROOT=/data/PSM;  EI=25; CO=25; MID=PSM;  EPOCHS=3 ;;
  MSL)  DATA=MSL;  ROOT=/data/MSL;  EI=55; CO=55; MID=MSL;  EPOCHS=10 ;;
  SMAP) DATA=SMAP; ROOT=/data/SMAP; EI=25; CO=25; MID=SMAP; EPOCHS=3 ;;
esac

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --model DLinear \
  --data "$DATA" \
  --root_path "$ROOT" \
  --model_id "$MID" \
  --features M \
  --seq_len 100 --pred_len 100 \
  --enc_in $EI --c_out $CO \
  --d_model 128 --d_ff 128 \
  --e_layers 3 \
  --anomaly_ratio 1 \
  --train_epochs $EPOCHS --batch_size 128 \
  --des "Exp_s${SEED}" --itr 1 \
  --seed $SEED
