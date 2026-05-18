#!/bin/bash
# Run custom MOEA on DTLZ2 (spherical Pareto front, 3 objectives)

cd /workspace

python deap/custom_moea.py \
    --problem dtlz2 \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-./output}
