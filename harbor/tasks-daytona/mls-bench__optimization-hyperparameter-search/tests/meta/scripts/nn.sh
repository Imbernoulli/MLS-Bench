#!/bin/bash
# Run HPO strategy on Neural Net tuning benchmark (Diabetes)

cd /workspace

python scikit-learn/custom_hpo.py \
    --benchmark nn \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-./output}
