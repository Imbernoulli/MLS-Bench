#!/bin/bash
# Rosenbrock function optimization, 30 dimensions
cd /workspace

python deap/custom_evolution.py \
    --function rosenbrock \
    --dim 30 \
    --pop-size 200 \
    --n-generations 500 \
    --cx-prob 0.9 \
    --mut-prob 0.2 \
    --seed ${SEED:-42}
