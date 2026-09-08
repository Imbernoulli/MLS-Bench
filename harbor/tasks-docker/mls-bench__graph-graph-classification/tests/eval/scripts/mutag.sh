#!/bin/bash
# Graph classification on MUTAG dataset (188 graphs, ~5 min on GPU)
cd /workspace
python pytorch-geometric/custom_graph_cls.py \
    --dataset MUTAG \
    --data-root /data/TUDataset \
    --hidden-dim 64 --num-layers 5 \
    --epochs 350 --batch-size 32 \
    --lr 0.01 --dropout 0.5 \
    --num-folds 10 \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-./output}
