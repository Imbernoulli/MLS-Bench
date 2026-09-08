#!/bin/bash
# Working directory is already /workspace (package root)

SEED=${SEED:-42}
# No official TSLib PatchTST anomaly detection script; aligned with classification PatchTST style

case "${ENV}" in
  PSM)  DATA=PSM;  ROOT=/data/PSM;  EI=25; CO=25; MID=PSM;  EPOCHS=3 ;;
  MSL)  DATA=MSL;  ROOT=/data/MSL;  EI=55; CO=55; MID=MSL;  EPOCHS=10 ;;
  SMAP) DATA=SMAP; ROOT=/data/SMAP; EI=25; CO=25; MID=SMAP; EPOCHS=3 ;;
esac

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --model PatchTST \
  --data "$DATA" \
  --root_path "$ROOT" \
  --model_id "$MID" \
  --features M \
  --seq_len 100 --pred_len 100 \
  --enc_in $EI --c_out $CO \
  --d_model 128 --d_ff 256 \
  --e_layers 3 --d_layers 1 \
  --n_heads 8 --dropout 0.1 \
  --anomaly_ratio 1 \
  --train_epochs $EPOCHS --batch_size 128 \
  --patience 3 --learning_rate 0.0001 \
  --des "Exp_s${SEED}" --itr 1 \
  --seed $SEED
