#!/bin/bash
set -e
cd /workspace/easy-few-shot-learning

# Create symlinks for CUB images and specs
mkdir -p data/CUB
ln -sfn /data/CUB/images data/CUB/images 2>/dev/null || true
# Symlink JSON specs from data directory if they exist there
for spec in train.json val.json test.json; do
    [ -f "/data/CUB/$spec" ] && ln -sfn "/data/CUB/$spec" "data/CUB/$spec" 2>/dev/null || true
done

ENV=CUB SEED=${SEED:-42} OUTPUT_DIR=${OUTPUT_DIR:-./output} \
    python -u custom_fewshot.py
