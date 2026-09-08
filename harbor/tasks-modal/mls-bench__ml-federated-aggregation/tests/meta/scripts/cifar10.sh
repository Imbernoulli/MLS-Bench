#!/bin/bash
# Run FL simulation on CIFAR-10 with Dirichlet non-IID split (alpha=0.1)

cd /workspace

python flower/custom_fl_aggregation.py \
    --dataset cifar10 \
    --data-dir /data \
    --num-clients 100 \
    --clients-per-round 10 \
    --num-rounds 200 \
    --local-epochs 5 \
    --local-lr 0.01 \
    --local-batch-size 64 \
    --dirichlet-alpha 0.1 \
    --eval-every 10 \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-./output}
