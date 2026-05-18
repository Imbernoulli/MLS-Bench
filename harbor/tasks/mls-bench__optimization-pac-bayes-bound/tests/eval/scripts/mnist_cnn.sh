#!/bin/bash
# PAC-Bayes bound optimization on MNIST with convolutional network (2-conv + 2-fc)
cd /workspace
OUT_DIR="${OUTPUT_DIR:-./output}/${ENV:-mnist-cnn}"

python PBB/custom_pac_bayes.py \
    --dataset mnist \
    --model cnn \
    --seed ${SEED:-42} \
    --batch-size 250 \
    --prior-epochs 10 \
    --posterior-epochs 20 \
    --prior-frac 0.5 \
    --delta 0.025 \
    --prior-sigma 0.03 \
    --mc-samples 50 \
    --output-dir "$OUT_DIR" \
    --data-dir /workspace/data
