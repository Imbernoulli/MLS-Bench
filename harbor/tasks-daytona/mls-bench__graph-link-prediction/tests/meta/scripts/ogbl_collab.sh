#!/bin/bash
# Link prediction on ogbl-collab collaboration network
cd /workspace

python pytorch-geometric-lp/custom_linkpred.py \
    --dataset ogbl-collab --data-dir /data \
    --hidden-channels 256 --num-layers 3 --dropout 0.0 \
    --lr 0.001 --weight-decay 0.0 \
    --epochs 400 --eval-every 10 --patience 30 \
    --batch-size 65536 \
    --seed ${SEED:-42} --output-dir "${OUTPUT_DIR:-./output}/${ENV:-ogbl-collab}"
