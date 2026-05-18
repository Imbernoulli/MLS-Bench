#!/bin/bash
# Run custom MOEA on ZDT1 (convex Pareto front, 2 objectives)

cd /workspace

python deap/custom_moea.py \
    --problem zdt1 \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-./output}
