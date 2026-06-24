#!/bin/bash
# Run custom MOEA on ZDT3 (disconnected Pareto front, 2 objectives)

cd /workspace

python "/tests/eval/_inputgen/apply.py" "optimization-multi-objective" /workspace
python deap/custom_moea.py \
    --problem zdt3 \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-./output}
