#!/bin/bash
# Active learning on OpenML spambase dataset (ID 44)
# 4601 samples, 57 features, 2 classes
cd /workspace
python badge/run_al.py \
    --did 44 \
    --alg ${ALG:-custom} \
    --seed ${SEED:-42} \
    --nStart 50 \
    --nQuery 50 \
    --nRounds 20 \
    --nEmb 128 \
    --lr 1e-3 \
    --output-dir ${OUTPUT_DIR:-.}
