#!/bin/bash
# Run FL simulation on Shakespeare (next character prediction, naturally non-IID)

cd /workspace

python flower/custom_fl_aggregation.py \
    --dataset shakespeare \
    --data-dir /data \
    --num-clients 100 \
    --clients-per-round 10 \
    --num-rounds 200 \
    --local-epochs 5 \
    --local-lr 0.01 \
    --local-batch-size 64 \
    --eval-every 10 \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-./output}
