#!/bin/bash
# Train MLP on Fashion-MNIST and evaluate calibration
cd /workspace
python scikit-learn/custom_calibration.py \
    --classifier mlp --dataset fashion_mnist \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR:-./output}
