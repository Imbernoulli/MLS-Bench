#!/bin/bash
# Working directory is already /workspace (package root)

SEED=${SEED:-42}
# Hyperparameters aligned with official TSLib per-dataset scripts

case "${ENV}" in
  EthanolConcentration)
    ROOT=/data/EthanolConcentration/; MID=EthanolConcentration
    DM=16; DFF=32; EL=2; EPOCHS=30; NK=6 ;;
  FaceDetection)
    ROOT=/data/FaceDetection/; MID=FaceDetection
    DM=64; DFF=256; EL=2; EPOCHS=30; NK=4 ;;
  Handwriting)
    ROOT=/data/Handwriting/; MID=Handwriting
    DM=32; DFF=64; EL=2; EPOCHS=30; NK=6 ;;
esac

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --model TimesNet \
  --data UEA \
  --root_path "$ROOT" \
  --model_id "$MID" \
  --d_model $DM --d_ff $DFF \
  --e_layers $EL --d_layers 1 \
  --top_k 3 --num_kernels $NK \
  --n_heads 16 --dropout 0.1 \
  --train_epochs $EPOCHS --batch_size 16 \
  --patience 10 --learning_rate 0.001 \
  --des "Exp_s${SEED}" --itr 1 \
  --seed $SEED
