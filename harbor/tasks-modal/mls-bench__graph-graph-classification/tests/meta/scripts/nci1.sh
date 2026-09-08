#!/bin/bash
# Graph classification on NCI1 dataset (4110 graphs, ~45 min on GPU)
cd /workspace
python pytorch-geometric/custom_graph_cls.py \
    --dataset NCI1 \
    --data-root /data/TUDataset \
    --hidden-dim 64 --num-layers 5 \
    --epochs 350 --batch-size 32 \
    --lr 0.01 --dropout 0.5 \
    --num-folds 10 \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-./output}
