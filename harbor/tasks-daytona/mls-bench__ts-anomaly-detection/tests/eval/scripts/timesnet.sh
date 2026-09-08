#!/bin/bash
# Working directory is already /workspace (package root)

SEED=${SEED:-42}
# Hyperparameters aligned with official TSLib per-dataset scripts

case "${ENV}" in
  PSM)
    DATA=PSM;  ROOT=/data/PSM;  EI=25; CO=25; MID=PSM
    DM=64; DFF=64; EL=2; EPOCHS=3 ;;
  MSL)
    DATA=MSL;  ROOT=/data/MSL;  EI=55; CO=55; MID=MSL
    DM=8; DFF=16; EL=1; EPOCHS=1 ;;
  SMAP)
    DATA=SMAP; ROOT=/data/SMAP; EI=25; CO=25; MID=SMAP
    DM=128; DFF=128; EL=3; EPOCHS=3 ;;
esac

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --model TimesNet \
  --data "$DATA" \
  --root_path "$ROOT" \
  --model_id "$MID" \
  --features M \
  --seq_len 100 --pred_len 0 \
  --enc_in $EI --c_out $CO \
  --d_model $DM --d_ff $DFF \
  --e_layers $EL \
  --top_k 3 \
  --anomaly_ratio 1 \
  --train_epochs $EPOCHS --batch_size 128 \
  --des "Exp_s${SEED}" --itr 1 \
  --seed $SEED
