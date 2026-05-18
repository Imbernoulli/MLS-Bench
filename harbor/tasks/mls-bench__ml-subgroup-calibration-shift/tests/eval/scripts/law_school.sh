#!/bin/bash
cd /workspace
python scikit-learn/custom_subgroup_calibration.py \
    --dataset law_school \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-./output}
