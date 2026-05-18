#!/bin/bash
python custom_pla.py \
    --test-set test2013 --data-dir /data \
    --epochs 800 --batch-size 128 --lr 1e-4 --patience 50 \
    --seed ${SEED:-42} --output-dir ${OUTPUT_DIR}
