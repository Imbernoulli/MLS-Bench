#!/bin/bash
# Rastrigin function optimization, 100 dimensions (scalability test)
cd /workspace

python deap/custom_evolution.py \
    --function rastrigin \
    --dim 100 \
    --pop-size 400 \
    --n-generations 1000 \
    --cx-prob 0.9 \
    --mut-prob 0.2 \
    --seed ${SEED:-42}
