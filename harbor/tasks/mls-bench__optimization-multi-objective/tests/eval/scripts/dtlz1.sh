#!/bin/bash
# Run custom MOEA on DTLZ1 (linear Pareto front with local fronts, 3 objectives)

cd /workspace

python deap/custom_moea.py \
    --problem dtlz1 \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-./output}
