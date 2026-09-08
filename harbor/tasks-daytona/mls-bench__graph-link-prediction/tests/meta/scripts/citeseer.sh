#!/bin/bash
# Link prediction on CiteSeer citation network
cd /workspace

python pytorch-geometric-lp/custom_linkpred.py \
    --dataset CiteSeer --data-dir /data \
    --hidden-channels 256 --num-layers 2 --dropout 0.0 \
    --lr 0.01 --epochs 200 --eval-every 10 --patience 20 \
    --seed ${SEED:-42} --output-dir "${OUTPUT_DIR:-./output}/${ENV:-CiteSeer}"
