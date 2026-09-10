#!/bin/bash
# Active learning on OpenML letter recognition dataset (ID 6)
# 20000 samples, 16 features, 26 classes
cd /workspace
python badge/run_al.py \
    --did 6 \
    --alg ${ALG:-custom} \
    --seed ${SEED:-42} \
    --nStart 100 \
    --nQuery 100 \
    --nRounds 20 \
    --nEmb 128 \
    --lr 1e-3 \
    --output-dir ${OUTPUT_DIR:-.}
