#!/bin/bash
# Active learning on OpenML splice dataset (ID 46)
# 3190 samples, 60 features, 3 classes
cd /workspace
python badge/run_al.py \
    --did 46 \
    --alg ${ALG:-custom} \
    --seed ${SEED:-42} \
    --nStart 50 \
    --nQuery 50 \
    --nRounds 20 \
    --nEmb 128 \
    --lr 1e-3 \
    --output-dir ${OUTPUT_DIR:-.}
